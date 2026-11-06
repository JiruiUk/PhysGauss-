"""Adapt frozen evaluator presentation in independent new files."""
from pathlib import Path

B=Path(__file__).resolve().parent
source=(B.parent/'RubberDuck_temporal_memory/evaluate.py').read_text()
source=source.replace('i10_temporal_memory import runtime as a','j7_long_memory import runtime as a')
source=source.replace("LABELS=['J6_parent'","LABELS=['J7_parent'").replace('J6_parent','J7_parent')
begin=source.index('def load_models():');end=source.index('\n\n@torch.no_grad()',begin)
source=source[:begin]+'''def load_models():
    from experiments.staged_reconstruction.RubberDuck_J7_long_memory.audit import audit
    p=read(a.BASE/'protocol.json');audits=audit()
    renderer,parent,_,_,_=r.load('rgb_control');models={'J7_parent':parent}
    for arm in p['arms']:
        renderer,scene,opts,_,_=r.load(arm)
        r.restore(a.RUNS/'formal'/arm/'checkpoint.pt',scene,opts);models[arm]=scene
    write(a.REPORT/'independent_audit.json',audits)
    return renderer,models,audits
'''+source[end:]
begin=source.index('@torch.no_grad()\ndef diagnostics');end=source.index('@torch.no_grad()\ndef track_diagnostic',begin)
source=source[:begin]+'''from experiments.staged_reconstruction.RubberDuck_J7_long_memory.diagnostics import diagnostics

'''+source[end:]
for name in ['MEMORIES','META','ASSET']:source=source.replace('a.'+name,'a.memory.'+name)
source=source.replace("a.OUT.mkdir(parents=True,exist_ok=True)","a.OUT.mkdir(parents=True,exist_ok=True);freeze()")
at=source.index('\ndef main():')
freeze='''
def freeze():
    files=[Path(__file__),a.BASE/'audit.py',a.BASE/'diagnostics.py',
        ROOT/'experiments/staged_reconstruction/RubberDuck_joint_pilot/evaluate.py',
        ROOT/'data/derived/kinematic_resplit/transforms_val.json']
    files += [DATA/(f['file_path']+'.png') for f in read(files[-1])['frames']]
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    path=a.REPORT/'evaluation_freeze.json'
    if path.exists():assert read(path)==hashes
    else:write(path,hashes)

'''
source=source[:at]+freeze+source[at:]
(B/'evaluate.py').write_text(source.replace('L4','J8'),encoding='utf-8')
print('J8 independent evaluator created')
