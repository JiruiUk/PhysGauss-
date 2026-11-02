"""Read-only compact status, no GPU allocation."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
state=json.loads((ROOT/'experiments/momentum_repair/progress.json').read_text())
run=ROOT/'runs/pidg_momentum_guard/M1_guarded_shared/rubber_duck'
for filename,key in [('momentum_gradient_guard.jsonl','last_guard'),('material_momentum.jsonl','last_momentum')]:
    p=run/filename
    if p.exists():state[key]=json.loads(p.read_text().splitlines()[-1])
print(json.dumps(state,ensure_ascii=False))
