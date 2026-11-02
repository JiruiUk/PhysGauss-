"""Compact read-only queue status for long training sessions."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
state = json.loads((ROOT/'experiments/flow_comparison/progress.json').read_text())
result = {'status': state['status'], 'active': state['active'], 'updated_utc': state['updated_utc'], 'error': state['error'], 'models': []}
for model in state['models']:
    row = {'id': model['id'], 'iteration': model['iteration']}
    val = model['latest_validation']
    if val:
        row['validation_step'] = val['iteration']; row['PSNR'] = val['metrics']['PSNR']
    run = ROOT/'runs/pidg_flow_comparison'/model['id']/'rubber_duck'
    for key, filename in [('flow', 'observed_flow.jsonl'), ('momentum', 'material_momentum.jsonl')]:
        p = run/filename
        if p.exists():
            row[key] = json.loads(p.read_text().splitlines()[-1])
    p = run/'kinematic_diagnostics.jsonl'
    if p.exists():
        value = json.loads(p.read_text().splitlines()[-1])
        row['velocity_relative_RMSE'] = sum(v['relative_rmse'] for v in value['times'])/len(value['times'])
    result['models'].append(row)
print(json.dumps(result, ensure_ascii=False))
