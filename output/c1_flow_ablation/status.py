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
queue=RUNS/'reuse_control_queue'
reused=(queue/'controller.json').exists()
controller_root=queue if reused else RUNS
result={'controller':read_json(controller_root/'controller.json') if (controller_root/'controller.json').exists() else {},'models':[]}
directories=([ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck',RUNS/'O1_C1_flow_on/rubber_duck']
    if reused else [RUNS/n/'rubber_duck' for n in NAMES])
for d in directories:
    name=d.parent.name; p=d/'checkpoints/resume.json'
    log=tail(d/'train.log',12000)
    matches=re.findall(r'Training progress[^\r\n]*?\|\s*(\d+)/50000',log)
    result['models'].append({'name':name,'historical_reused':reused and name=='C1_repaired','saved_iteration':read_json(p)['iteration'] if p.exists() else 0,
        'live_iteration':int(matches[-1]) if matches else None,'log_tail':log[-1300:]})
print(json.dumps(result,ensure_ascii=False,indent=2))
