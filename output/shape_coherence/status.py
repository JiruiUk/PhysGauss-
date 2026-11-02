"""Read-only shape experiment state; never starts GPU work."""
from pathlib import Path
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.metric_tuning import run as engine
from variants.shape_coherence.run import RUNS,REPORTS,NAME


def main():
    result={}
    for filename in ['controller.json','active.json']:
        p=RUNS/filename
        if p.exists():result[filename]=engine.read_json(p)
    for name in ['_flags_off','_smoke_P1',NAME]:
        directory=RUNS/name/'rubber_duck';receipt=directory/'checkpoints/resume.json'
        if not directory.exists():continue
        info={'checkpoint':engine.read_json(receipt) if receipt.exists() else None}
        for log in ['train.log','shape_loss.jsonl','camera_sampling.jsonl']:
            p=directory/log
            if p.exists():
                with p.open('rb') as stream:
                    stream.seek(max(0,p.stat().st_size-5000));raw=stream.read().decode('utf8',errors='replace')
                info[log]={'modified':p.stat().st_mtime,'bytes':p.stat().st_size,'tail':raw[-2000:]}
        result[name]=info
    for filename in ['preflight.json','candidate_graph_review.json','selection_lock.json','final_verification.json']:
        p=(RUNS if filename=='selection_lock.json' else REPORTS)/filename
        if p.exists():result[filename]=engine.read_json(p)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
