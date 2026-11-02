from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
file=ROOT/'experiments/coherent_momentum/progress.json'
print(file.read_text(encoding='utf8') if file.exists() else 'not started')
run=ROOT/'runs/pidg_coherent_momentum/N1_coherent_momentum/rubber_duck'
for name in ['coherent_momentum.jsonl','momentum_gradient_guard.jsonl']:
    p=run/name
    if p.exists():
        lines=p.read_text(encoding='utf8').splitlines()
        print(name,'records',len(lines),'last',lines[-1] if lines else '')
