"""Verify every completed trial, paired inputs/noise and first candidate outputs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import zipfile

out=Path(__file__).resolve().parent
finished=json.loads((out/'finished.json').read_text())
assert finished['trials']==6 and finished['episodes']==150
selection=json.loads((out/'task-selection.json').read_text())
expected={f"task-{r['task_id']:04d}-episode-0000" for r in selection['tasks']}
report={'experiment':json.loads((out/'experiment.json').read_text()),'finished':finished,
        'trials':{},'aggregate':{},'paired_checks':{},'paired_outcomes':{}}
all_calls={};initial={};records={}

def distribution(values):
    values=sorted(values)
    def quantile(p):
        position=(len(values)-1)*p;index=int(position)
        return values[index]+(values[min(index+1,len(values)-1)]-values[index])*(position-index)
    return {'count':len(values),'mean':statistics.mean(values),'p50':quantile(.5),
            'p95':quantile(.95),'max':max(values)}


for summary in json.loads((out/'summary.json').read_text()):
    assert summary['exit_code']==0
    name=summary['name'];trial=out/name
    data=json.loads((trial/'evaluation/result.json').read_text())
    assert data['result']['state']=='complete' and data['result']['episodes']==25
    assert data['runtime']['concurrency']==4 and data['runtime']['replicas']==1
    rows=[]
    for f in (trial/'evaluation/queue/completed').glob('*.json'):
        rows.extend(json.loads(f.read_text())['records'])
    assert len(rows)==25 and {r['record_id'] for r in rows}==expected
    assert all(r['metadata']['episode_index']==0 for r in rows)
    logs=[json.loads(line) for line in (trial/'timing.jsonl').read_text().splitlines()]
    calls=[r for r in logs if r['type']=='infer']
    ends=[r for r in logs if r['type']=='episode_end']
    assert len(ends)==25 and {r['episode_id'] for r in ends}==expected
    assert {r['episode_id'] for r in calls}==expected
    assert all(r['mode']==summary['mode'] and r['seed']==summary['seed'] for r in calls)
    inputs=[json.loads(line) for line in (trial/'initial-inputs.jsonl').read_text().splitlines()]
    assert len(inputs)==25 and {r['episode_id'] for r in inputs}==expected
    initial[name]={r['episode_id']:r for r in inputs}
    records[name]={r['record_id']:r for r in rows}
    all_calls[name]=calls
    errors=trial/'evaluation/errors.jsonl'
    assert not errors.exists() or not errors.read_text().strip()
    report['trials'][name]={**summary,'policy_ms':distribution([r['act_wall_ms'] for r in calls]),
        'fallbacks':sum(r['fallback'] for r in calls),'cold_score_calls':sum(r.get('cold_generation',False) or r.get('cold_fallback',False) for r in calls),
        'initial_inputs':inputs,'records':rows,
        'first_actions':[r for r in calls if r['step']==0],
        'failed_steps':dict(Counter(r['steps'] for r in rows if not r['success']))}

for mode in ('ten','four','scored'):
    names=[f'seed{seed}-{mode}' for seed in (7,17)]
    calls=[r for name in names for r in all_calls[name]]
    rows=[r for name in names for r in records[name].values()]
    summaries=[report['trials'][name] for name in names]
    successes=sum(r['success'] for r in rows)
    report['aggregate'][mode]={'episodes':len(rows),'unique_tasks':25,'noise_repetitions':2,
        'successes':successes,'success_rate':successes/len(rows),
        'successes_by_noise_seed':{str(seed):report['trials'][f'seed{seed}-{mode}']['result']['successes'] for seed in (7,17)},
        'policy_ms':distribution([r['act_wall_ms'] for r in calls]),
        'evaluation_s_sum':sum(r['phases']['evaluation_s'] for r in summaries),
        'total_wall_s_sum':sum(r['total_wall_s'] for r in summaries),
        'gpu_peak_mib':max(max(r['gpu']['gpu_mem_mib_max'].values()) for r in summaries),
        'fallbacks':sum(r['fallback'] for r in calls),'fallback_rate':sum(r['fallback'] for r in calls)/len(calls)}

for seed in (7,17):
    names=[f'seed{seed}-{mode}' for mode in ('ten','four','scored')]
    common_inputs=[];noise_matches=[];prefix_deltas=[]
    for episode in sorted(expected):
        inputs=[initial[name][episode] for name in names]
        common_inputs.append(all(r['fingerprints']==inputs[0]['fingerprints'] and r['prompt']==inputs[0]['prompt'] for r in inputs[1:]))
        first=[next(r for r in all_calls[name] if r['episode_id']==episode and r['step']==0) for name in names]
        noise_matches.append(all(r['noise_seed']==first[0]['noise_seed'] and r['noise_a_sha256']==first[0]['noise_a_sha256'] for r in first[1:]))
        a=first[1]['candidate_a_prefix'];b=first[2]['candidate_a_prefix']
        diffs=[abs(x-y) for row_a,row_b in zip(a,b) for x,y in zip(row_a,row_b)]
        prefix_deltas.append({'episode_id':episode,'max_abs':max(diffs),'mean_abs':statistics.mean(diffs)})
    stepmaps=[{(r['episode_id'],r['step']):r['noise_seed'] for r in all_calls[name]} for name in names]
    shared_steps=set(stepmaps[0])&set(stepmaps[1])&set(stepmaps[2])
    equal_steps=sum(stepmaps[0][k]==stepmaps[1][k]==stepmaps[2][k] for k in shared_steps)
    report['paired_checks'][str(seed)]={'matched_initial_inputs':sum(common_inputs),'total':25,
        'matched_initial_candidate_a_noise':sum(noise_matches),
        'common_prediction_steps':len(shared_steps),'equal_noise_at_common_steps':equal_steps,
        'official4_vs_shared_candidate_a':prefix_deltas,
        'official4_vs_shared_a_global_max_abs':max(r['max_abs'] for r in prefix_deltas)}
    assert all(common_inputs) and all(noise_matches) and equal_steps==len(shared_steps)
    comparisons={}
    for reference,treatment in (('ten','four'),('four','scored'),('ten','scored')):
        a=records[f'seed{seed}-{reference}'];b=records[f'seed{seed}-{treatment}']
        comparisons[reference+'->'+treatment]={'failure_to_success':sum(not a[k]['success'] and b[k]['success'] for k in expected),
            'success_to_failure':sum(a[k]['success'] and not b[k]['success'] for k in expected)}
    report['paired_outcomes'][str(seed)]=comparisons

report['gpu_processes_after']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
report['gpu_after']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
report['upstream_status']={name:subprocess.check_output(['/root/prp/kit-tools/bin/git','-C','/root/prp/'+name,'status','--short'],text=True).strip() for name in ('openpi','RoboScope-Kit')}
(out/'ablation-results.json').write_text(json.dumps(report,indent=2))
archive=out/'ablation-evidence.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for name in ('ablation-results.json','experiment.json','task-selection.json','summary.json','finished.json','ablation_policy.py','ablation_cli.py','ablation_run.py','ablation_summarize.py'):
        z.write(out/name,name)
    for name in report['trials']:
        for file in ('evaluation/result.json','initial-inputs.jsonl','timing.jsonl','command.json','policy-config.json'):
            z.write(out/name/file,name+'/'+file)
print('AGGREGATE',json.dumps(report['aggregate']))
print('PAIR_CHECKS',json.dumps({seed:{k:v for k,v in row.items() if k!='official4_vs_shared_candidate_a'} for seed,row in report['paired_checks'].items()}))
print('PAIRED_OUTCOMES',json.dumps(report['paired_outcomes']))
print('GPU_AFTER',report['gpu_after'],'PROCESSES',repr(report['gpu_processes_after']))
print('ARCHIVE',archive,'SHA256',hashlib.sha256(archive.read_bytes()).hexdigest())
