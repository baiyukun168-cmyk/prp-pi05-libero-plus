"""Validate four complete Camera result sets and compute pooled success rate."""

import argparse
import json
from pathlib import Path
import re


def combine(runs,metadata):
    expected_suites={"libero_spatial":376,"libero_object":396,"libero_goal":408,"libero_10":419}
    rows={}
    for run in runs:
        result=json.loads((run/"evaluation/result.json").read_text())
        suite=result["benchmark"]["suite"]
        if suite not in expected_suites or suite in rows:
            raise ValueError("Unexpected or duplicate suite")
        expected={int(item["id"])-1 for item in metadata[suite] if item["category"]=="Camera Viewpoints"}
        actual=set(result["benchmark"]["task_ids"])
        records=result["record_success"]
        if len(expected)!=expected_suites[suite] or actual!=expected or len(records)!=len(expected):
            raise ValueError(f"Incomplete Camera population in {suite}")
        record_ids=set()
        for name,value in records.items():
            match=re.fullmatch(r"task-(\d+)-episode-(\d+)",name)
            if match is None or int(match[2])!=0 or not isinstance(value,bool):
                raise ValueError(f"Unexpected terminal record {name}")
            record_ids.add(int(match[1]))
        count=result["result"]
        if record_ids!=expected or count["state"]!="complete" or count["episodes"]!=len(expected):
            raise ValueError(f"Terminal accounting incomplete in {suite}")
        successes=sum(records.values())
        if successes!=count["successes"]:
            raise ValueError("Success counts differ from terminal records")
        rows[suite]={"episodes":len(expected),"successes":successes,"success_rate":successes/len(expected),"run_dir":str(run)}
    if set(rows)!=set(expected_suites):
        raise ValueError("All four suites are required for the final Camera rate")
    total=sum(row["episodes"] for row in rows.values())
    successes=sum(row["successes"] for row in rows.values())
    return {"category":"Camera Viewpoints","episodes":total,"successes":successes,
            "failures":total-successes,"success_rate":successes/total,
            "aggregation":"All 1599 individual Camera entries pooled; one episode per entry","suites":rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,nargs=4,required=True)
    parser.add_argument("--metadata",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=combine(args.runs,json.loads(args.metadata.read_text()))
    with args.output.open("x") as stream:
        stream.write(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
