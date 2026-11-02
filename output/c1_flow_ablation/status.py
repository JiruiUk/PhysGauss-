"""Read-only status; never launches training or consumes CUDA memory."""
import json
from pathlib import Path
import re
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from variants.c1_flow_ablation.run import RUNS,NAMES,REPORTS
from variants.metric_tuning.run import read_json,tail
result={'controller':read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {},'models':[]}
for name in NAMES:
    d=RUNS/name/'rubber_duck'; p=d/'checkpoints/resume.json'
    log=tail(d/'train.log',12000)
    matches=re.findall(r'Training progress[^\r\n]*?\|\s*(\d+)/50000',log)
    result['models'].append({'name':name,'saved_iteration':read_json(p)['iteration'] if p.exists() else 0,
        'live_iteration':int(matches[-1]) if matches else None,'log_tail':log[-1300:]})
print(json.dumps(result,ensure_ascii=False,indent=2))
