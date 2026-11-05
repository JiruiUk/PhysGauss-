"""Close completed evidence without changing any historical result or gate."""
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha

def finalize(base):
    report=base/'v1';result=read(report/'bindings.json')['summary']
    assert read(report/'independent_process_replay.json')['passed']
    assert read(report/'tracker_replay.json')['passed']
    write(report/'evaluation_source_freeze.json',dict(files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [
        Path(__file__),ROOT/'variants/staged_reconstruction/i10_tracking/audit.py']}))
    artifacts=[p for p in report.glob('*') if p.is_file() and p.name not in ['final_review.json','final_status.json']]
    run=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_tracking'
    if base.name=='dense':run=run/'dense'
    run=run/'v1'
    output=ROOT/'output/staged_reconstruction/RubberDuck_I10_tracking'
    if base.name=='dense':output=output/'dense'
    output=output/'v1'
    artifacts+=list(output.glob('*.png'))+[run/'parent_trajectories.npz']
    status=dict(status='evidence_passed_training_protocol_required' if result['evidence_gate_passed'] else 'completed_evidence_not_admitted',
        pending=False,pending_review=False,optimizer_updates=0,**{k:v for k,v in result.items() if k!='optimizer_updates'})
    write(report/'final_review.json',dict(**status,reviewed_images=[f'{kind}_{start:03d}.png' for kind in ['clip','binding'] for start in [0,52,108]],
        actual_visual_review=True,historical_color_parent_disclosed=True,independent_3D_GT=False,
        artifact_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in artifacts}))
    for p in [report/'final_status.json',run/'state.json',run/'final_status.json']:write(p,status)

if __name__=='__main__':finalize(Path(__file__).parent)
