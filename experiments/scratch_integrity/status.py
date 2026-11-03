"""Compact local progress snapshot; no model imports or predictions."""
from pathlib import Path
import json
import re
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];RUNS=ROOT/'runs/pidg_scratch_integrity'


def read(p):return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def main():
    state=read(RUNS/'controller.json');active=read(RUNS/'active.json');result={'UTC':datetime.now(timezone.utc).isoformat(),'controller':state}
    if active:
        directory=Path(active['run']);log=directory/'train.log'
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-24000));text=stream.read().decode('utf-8',errors='replace')
        matches=re.findall(r'(\d+)/50001[^\n\r]*',text)
        result['active']={k:active[k] for k in ['id','phase','pid']}
        if matches:result['active']['latest_progress_iteration']=int(matches[-1])
        receipt=read(directory/'checkpoints/resume.json')
        if receipt:result['active']['last_saved_iteration']=receipt['iteration']
        history=directory/'metric_history.jsonl'
        if history.exists():result['active']['latest_validation']=json.loads(history.read_text().splitlines()[-1])
    result['pipeline']=read(RUNS/'pipeline.json')
    for n in ['S0_scratch_control','S1_scratch_transfer']:
        p=RUNS/'pilot'/n/'rubber_duck/checkpoints/resume.json'
        if p.exists():result[n]=read(p)['iteration']
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
