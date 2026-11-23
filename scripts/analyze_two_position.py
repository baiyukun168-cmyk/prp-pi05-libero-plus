"""Audit baseline/speculative archives without extracting untrusted paths.

Usage: python scripts/analyze_two_position.py BASELINE.tar.gz SPEC.tar.gz OUT_DIR
Uses the Python standard library; raw per-episode evidence stays in OUT_DIR.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile

CATEGORIES = ['camera', 'robot', 'language', 'light', 'background', 'noise', 'layout']
SUITES = ['libero_10', 'libero_spatial', 'libero_object', 'libero_goal']


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def load_archive(path, speculative):
    runs = {}
    sources = {}
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = member.name.split('/')
            if speculative:
                parts = parts[1:]
                if parts and parts[0] == 'resume-20261123':
                    parts = parts[1:]
                    segment = 'resumed'
                else:
                    segment = 'original'
                if member.name.endswith('/spec.py'):
                    sources['sampler'] = archive.extractfile(member).read().decode()
            else:
                segment = 'baseline'
            if len(parts) < 3 or parts[0] not in CATEGORIES or parts[1] not in SUITES:
                continue
            cat, suite = parts[:2]
            run = runs.setdefault((cat, suite, segment), {'records': {}, 'timing': {}, 'timing_end': set()})
            suffix = '/'.join(parts[2:])
            wanted = suffix in ['timing.jsonl', 'task-selection.json', 'command.json',
                                'evaluation/result.json', 'evaluation/metrics.json']
            if not wanted and not suffix.startswith('evaluation/queue/completed/'):
                continue
            stream = archive.extractfile(member)
            if suffix == 'timing.jsonl':
                pending_spec = None
                for line in stream:
                    row = json.loads(line)
                    if row.get('type') == 'spec':
                        pending_spec = row
                    elif row.get('type') == 'infer':
                        e = run['timing'].setdefault(row['episode_id'], {'act': [], 'predict': [], 'accepted': 0, 'attempts': 0})
                        assert row['generation_steps'] == 10 and row['execution_steps'] == 5
                        e['act'].append(row['act_wall_ms'])
                        e['predict'].append(row['predict_ms'])
                        if speculative:
                            assert pending_spec is not None
                            assert pending_spec['threshold'] == 0.02
                            e['accepted'] += pending_spec['accepted']
                            e['attempts'] += pending_spec['attempts']
                            pending_spec = None
                    elif row.get('type') == 'episode_end':
                        run['timing_end'].add(row['episode_id'])
            elif suffix.startswith('evaluation/queue/completed/'):
                item = json.load(stream)
                for record in item['records']:
                    assert record['metadata']['episode_index'] == 0
                    rid = record['record_id']
                    assert rid not in run['records'], rid
                    run['records'][rid] = record
            else:
                run[suffix] = json.load(stream)
    return runs, sources


def summarize(runs):
    output, episodes = {}, {}
    for (cat, suite, segment), run in runs.items():
        command = run['command.json']
        expected_workers = 8 if cat in ('noise', 'layout') else 4
        assert int(command[command.index('--concurrency') + 1]) == expected_workers
        assert int(command[command.index('--replicas') + 1]) == 1
        selection = run['task-selection.json']
        selected = {f"task-{r['task_id']:04d}-episode-0000" for r in selection['tasks']}
        records = run['records']
        assert set(records) <= selected
        assert set(records) <= run['timing_end'], (cat, suite, 'missing episode_end')
        result = run.get('evaluation/result.json')
        if result:
            assert set(records) == selected
            assert result['result']['state'] == 'complete'
            assert result['result']['episodes'] == len(records)
            assert result['result']['successes'] == sum(r['success'] for r in records.values())
            assert not result['recovery']['worker_failures'] and not result['recovery']['oom_events']
            phase = result['phase_timings']
            peak = result['resources']['gpu']['peak_memory_bytes'] / 2**30
        else:
            assert (cat, suite, segment) == ('noise', 'libero_10', 'original')
            assert len(records) == 359
            phase = run['evaluation/metrics.json']
            peak = None
        d = output.setdefault(cat, {'act': [], 'predict': [], 'episode_mean': [], 'successes': 0,
                                   'episodes': 0, 'steps': 0, 'evaluation_s': 0, 'load_s': 0,
                                   'runtime_prepare_s': 0, 'gpu_peak_gib': 0, 'missing_gpu_segments': [],
                                   'accepted': 0, 'attempts': 0, 'segments': []})
        d['evaluation_s'] += phase['evaluation_s']
        d['load_s'] += phase['policy_load_s']
        d['runtime_prepare_s'] += phase['runtime_prepare_s']
        if peak is None:
            d['missing_gpu_segments'].append(f'{cat}/{suite}/{segment}')
        else:
            d['gpu_peak_gib'] = max(d['gpu_peak_gib'], peak)
        d['segments'].append({'suite': suite, 'segment': segment, 'episodes': len(records),
                              'successes': sum(r['success'] for r in records.values()),
                              'evaluation_s': phase['evaluation_s'], 'gpu_peak_gib': peak})
        for rid, record in records.items():
            key = (cat, suite, rid)
            assert key not in episodes, key
            timing = run['timing'][rid]
            assert timing['act'] and all(math.isfinite(v) and v > 0 for v in timing['act'])
            task_id = record['metadata']['task_id']
            prompt = selection['prompts'][rid]
            task = next(t for t in selection['tasks'] if t['task_id'] == task_id)
            episodes[key] = {'success': bool(record['success']), 'steps': record['steps'],
                             'elapsed_s': record['elapsed_s'], 'mean_ms': statistics.mean(timing['act']),
                             'prompt': prompt, 'task': task}
            d['act'].extend(timing['act'])
            d['predict'].extend(timing['predict'])
            d['episode_mean'].append(statistics.mean(timing['act']))
            d['accepted'] += timing['accepted']
            d['attempts'] += timing['attempts']
            d['successes'] += record['success']
            d['episodes'] += 1
            d['steps'] += record['steps']
    assert len(episodes) == 10030
    for cat, d in output.items():
        d['success_rate_pct'] = 100 * d['successes'] / d['episodes']
        d['call_count'] = len(d['act'])
        d['mean_ms'] = statistics.mean(d['act'])
        d['p50_ms'] = percentile(d['act'], .5)
        d['p95_ms'] = percentile(d['act'], .95)
        d['predict_mean_ms'] = statistics.mean(d['predict'])
        d['preprocess_and_other_mean_ms'] = statistics.mean(a-b for a,b in zip(d['act'],d['predict']))
        d['episode_weighted_mean_ms'] = statistics.mean(d['episode_mean'])
        d['episode_per_min'] = 60 * d['episodes'] / d['evaluation_s']
        d['active_total_s'] = d['evaluation_s'] + d['load_s'] + d['runtime_prepare_s']
        d['acceptance_pct'] = 100 * d['accepted'] / d['attempts'] if d['attempts'] else None
        for field in ['act','predict','episode_mean']:
            del d[field]
    return output, episodes


def main():
    baseline_path, spec_path, output_path = map(Path, sys.argv[1:])
    output_path.mkdir(parents=True, exist_ok=True)
    baseline_runs, _ = load_archive(baseline_path, False)
    print('Loaded baseline', flush=True)
    spec_runs, sources = load_archive(spec_path, True)
    print('Loaded speculative', flush=True)
    baseline, b = summarize(baseline_runs)
    spec, s = summarize(spec_runs)
    assert set(b) == set(s), 'Task mismatch'
    changed = []
    for key in b:
        assert b[key]['prompt'] == s[key]['prompt'], ('prompt', key)
        assert b[key]['task'] == s[key]['task'], ('task metadata', key)
        if b[key]['success'] != s[key]['success']:
            changed.append(key)
    rows = []
    for cat in CATEGORIES:
        old, new = baseline[cat], spec[cat]
        wins = sum(not b[k]['success'] and s[k]['success'] for k in changed if k[0] == cat)
        losses = sum(b[k]['success'] and not s[k]['success'] for k in changed if k[0] == cat)
        n = wins + losses
        pval = min(1., 2 * sum(math.comb(n, j) for j in range(min(wins, losses)+1)) / 2**n) if n else 1.
        row = {'dimension': cat, 'episodes': old['episodes'], 'baseline_success': old['successes'],
               'speculative_success': new['successes'], 'baseline_success_pct': old['success_rate_pct'],
               'speculative_success_pct': new['success_rate_pct'],
               'success_delta_pp': new['success_rate_pct']-old['success_rate_pct'],
               'paired_wins': wins, 'paired_losses': losses, 'exploratory_mcnemar_exact_p': pval,
               'baseline_mean_ms': old['mean_ms'], 'speculative_mean_ms': new['mean_ms'],
               'latency_reduction_pct': 100*(1-new['mean_ms']/old['mean_ms']),
               'call_speedup': old['mean_ms']/new['mean_ms'],
               'baseline_p95_ms': old['p95_ms'], 'speculative_p95_ms': new['p95_ms'],
               'baseline_gpu_peak_gib': old['gpu_peak_gib'], 'speculative_gpu_peak_gib': new['gpu_peak_gib'],
               'speculative_gpu_peak_incomplete': bool(new['missing_gpu_segments']),
               'baseline_evaluation_s': old['evaluation_s'], 'speculative_evaluation_s': new['evaluation_s'],
               'evaluation_speedup': old['evaluation_s']/new['evaluation_s'],
               'baseline_steps': old['steps'], 'speculative_steps': new['steps'],
               'acceptance_pct': new['acceptance_pct']}
        rows.append(row)
    overview = {'baseline_success': sum(x['successes'] for x in baseline.values()),
                'speculative_success': sum(x['successes'] for x in spec.values()), 'episodes': len(b),
                'baseline_mean_ms': sum(x['mean_ms']*x['call_count'] for x in baseline.values())/sum(x['call_count'] for x in baseline.values()),
                'speculative_mean_ms': sum(x['mean_ms']*x['call_count'] for x in spec.values())/sum(x['call_count'] for x in spec.values()),
                'baseline_evaluation_s': sum(x['evaluation_s'] for x in baseline.values()),
                'speculative_evaluation_s': sum(x['evaluation_s'] for x in spec.values()),
                'baseline_active_total_s': sum(x['active_total_s'] for x in baseline.values()),
                'speculative_active_total_s': sum(x['active_total_s'] for x in spec.values()),
                'baseline_gpu_peak_gib': max(x['gpu_peak_gib'] for x in baseline.values()),
                'speculative_gpu_peak_gib_lower_bound': max(x['gpu_peak_gib'] for x in spec.values()),
                'paired_wins': sum(r['paired_wins'] for r in rows), 'paired_losses': sum(r['paired_losses'] for r in rows)}
    overview['latency_reduction_pct'] = 100*(1-overview['speculative_mean_ms']/overview['baseline_mean_ms'])
    overview['evaluation_speedup'] = overview['baseline_evaluation_s']/overview['speculative_evaluation_s']
    summary = {'protocol': {'gpu':'RTX 4090', 'precision':'JAX BF16','generation_grid_steps':10,
                            'execution_steps':5,'seed':7,'episodes_per_task_variant':1,
                            'workers':'4 for first five dimensions; 8 for Noise/Layout','replicas':1,
                            'speculative_threshold':0.02,'sampler_sha256':hashlib.sha256(sources['sampler'].encode()).hexdigest()},
               'verification': {'paired_unique_episodes':len(b),'identical_task_and_prompt_metadata':True,
                                'completed_episode_end_present':True,'finished_segments_without_worker_failure_or_oom':True,
                                'partial_noise_359_records_retained':True,'unfinished_pre_shutdown_calls_excluded':True},
               'overview':overview,'comparison':rows,'baseline':baseline,'speculative':spec,
               'archive_sha256':{'baseline':hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                                 'speculative':hashlib.sha256(spec_path.read_bytes()).hexdigest()}}
    (output_path/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (output_path/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (output_path/'paired_episodes.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['dimension','suite','episode_id','baseline_success','speculative_success','baseline_steps','speculative_steps','baseline_mean_ms','speculative_mean_ms'])
        for k in sorted(b):w.writerow([*k,b[k]['success'],s[k]['success'],b[k]['steps'],s[k]['steps'],b[k]['mean_ms'],s[k]['mean_ms']])
    (output_path/'two_position_policy.py').write_text(sources['sampler'],encoding='utf-8')
    print(json.dumps({'overview':overview,'comparison':rows},indent=2))


if __name__ == '__main__':
    main()
