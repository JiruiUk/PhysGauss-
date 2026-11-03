"""Freeze D7 train-only parents/code; no evaluator assets enter inference."""
import argparse
from pathlib import Path
from ..prepare_data import ROOT,read,write,sha

BASE=ROOT/'experiments/staged_reconstruction/S1_plane_transport'
REPORT=BASE/'v1'
RUNS=ROOT/'runs/pidg_staged_reconstruction/S1_plane_transport/v1'
PARENT=ROOT/'runs/pidg_staged_reconstruction/S1_rgb_plane_warp/v1'
SOURCE=ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2'


def stopped():
    return any(p.exists() for p in [BASE/'stop.request',RUNS/'stop.request',ROOT/'stop.request',
        ROOT/'runs/pidg_staged_reconstruction/stop.request'])


def verify():
    for name,h in read(REPORT/'source_freeze.json').items(): assert sha(ROOT/name)==h,name


def main():
    p=argparse.ArgumentParser(); p.add_argument('--freeze',action='store_true'); args=p.parse_args()
    assert args.freeze and not stopped() and not (REPORT/'source_freeze.json').exists()
    protocol=read(BASE/'protocol.json'); prereg=read(BASE/'preregistration.json')
    for name,h in prereg['files'].items(): assert sha(ROOT/name)==h,name
    tests=(REPORT/'tests.log').read_text(); assert 'passed' in tests and 'failed' not in tests and 'ERROR' not in tests
    parent_freeze=ROOT/'experiments/staged_reconstruction/S1_rgb_plane_warp/v1/source_freeze.json'
    inputs=dict(read(parent_freeze))
    for name,h in inputs.items(): assert sha(ROOT/name)==h,name
    inputs={**inputs}; paths=list(Path(__file__).parent.glob('*.py'))+[BASE/'plan.md',BASE/'protocol.json',BASE/'preregistration.json',
        ROOT/'tests/test_plane_transport.py',REPORT/'tests.log',
        ROOT/'variants/staged_reconstruction/adaptive_static/geometry.py',
        ROOT/'variants/staged_reconstruction/rgb_prior_proposals/core.py',ROOT/'variants/staged_reconstruction/rgb_prior_proposals/io.py',
        ROOT/'variants/staged_reconstruction/synthetic_audit/input.py',ROOT/'variants/staged_reconstruction/prepare_data.py',
        ROOT/'variants/staged_reconstruction/pretrained_depth/model.py',
        ROOT/'variants/staged_reconstruction/rgb_plane_warp/core.py']
    for name,h in protocol['parent_receipts_sha256'].items():
        assert sha(ROOT/name)==h; paths.append(ROOT/name); inputs.update(read(ROOT/name)['outputs'])
    for case in protocol['synthetic_cases']:
        train=ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train'
        paths += [train/'cameras.json',SOURCE/f'{case}_inference.json']+list((train/'rgb').glob('*.png'))
    inputs.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    assert not any('/evaluation_only/' in name or 'oracle' in name for name in inputs)
    # Gate objects must remain exactly the earlier rejected D4 protocol.
    old=read(ROOT/'experiments/staged_reconstruction/S1_rgb_prior_proposals/protocol.json')
    for key in ['numeric_improvement_gate_both_cases','follow_on_Gaussian_protocol_evidence_gate_both_cases']:
        assert protocol[key]==old[key]
    for name,h in inputs.items(): assert sha(ROOT/name)==h,name
    write(REPORT/'source_freeze.json',inputs)
    print('PLANE_TRANSPORT_FROZEN',len(inputs),sha(REPORT/'source_freeze.json'),flush=True)


if __name__=='__main__': main()
