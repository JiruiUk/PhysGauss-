"""Independent source/input, confidence, optimizer and full sampler-state audit."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from ..prepare_data import ROOT, read, write, sha
from .prepare import REPORT, RUNS, verify
from .train import confidence_hash


def freeze():
    assert not (REPORT/'source_freeze.json').exists(), 'Do not replace a frozen experiment'
    frozen = read(ROOT/'experiments/staged_reconstruction/S1_automatic/source_freeze.json')
    for path, expected in frozen.items(): assert sha(ROOT/path) == expected, path
    files = list(Path(__file__).parent.glob('*.py'))+[ROOT/'tests/test_cross_temporal_static.py',
        REPORT/'protocol.json', REPORT/'discovery.json', REPORT/'candidate_review.json', REPORT/'test_receipt.json',
        RUNS/'initialization.npz', RUNS/'initial_confidence.npz', RUNS/'geometric_confidence.npz']
    for p in files: frozen[p.relative_to(ROOT).as_posix()] = sha(p)
    old = read(ROOT/'experiments/staged_reconstruction/completion_integrity.json')['final_checkpoints']
    for row in old: frozen[row['path']] = row['sha256']
    for name, expected in [
        ('S1_automatic/formal/checkpoint.pt', read(ROOT/'experiments/staged_reconstruction/S1_automatic/formal_state_audit.json')['checkpoint_sha256']),
        ('S3_union/physics/formal/checkpoint.pt', read(ROOT/'experiments/staged_reconstruction/S3_union/physics_state_audit.json')['checkpoint_sha256'])]:
        frozen['runs/pidg_staged_reconstruction/'+name] = expected
    write(REPORT/'source_freeze.json', frozen); verify()


def audit(arm, smoke, n):
    directory = RUNS/('smoke_v2' if smoke else 'formal')/arm
    path = directory/'checkpoint.pt'; state = torch.load(path, map_location='cpu', weights_only=False)
    initial = torch.load(directory/'initial.pt', map_location='cpu', weights_only=False)
    assert state['iteration'] == n and sha(path) == read(directory/'checkpoint.json')['sha256']
    assert state['protocol_sha256'] == initial['protocol_sha256'] == sha(REPORT/'protocol.json')
    assert state['source_manifest_sha256'] == sha(REPORT/'source_freeze.json')
    assert state['arm'] == arm and state['static_only'] and state['dynamic_parameters'] == 0
    assert all(v.isfinite().all() for v in state['model'].values())
    counters = [int(v['step']) for v in state['optimizer']['state'].values()]
    assert len(counters) == 6 and all(v == n for v in counters)
    assert all(v.isfinite().all() for entry in state['optimizer']['state'].values() for v in entry.values() if torch.is_tensor(v))
    count = len(read(directory/'sampling_contract.json')['frames'])
    rng = random.Random(); rng.setstate(initial['rng_python']); stack, sequence = [], []
    for _ in range(n):
        if not stack: stack = rng.sample(range(count), count)
        sequence.append(stack.pop())
    actual = [json.loads(line) for line in (directory/'sampling.jsonl').read_text().splitlines()]
    assert [r['step'] for r in actual] == list(range(1, n+1))
    assert [r['frame_index'] for r in actual] == sequence
    assert rng.getstate() == state['rng_python'] and stack == state['camera_stack']
    assert np.array_equal(initial['rng_numpy'][1], state['rng_numpy'][1])
    assert initial['rng_numpy'][0] == state['rng_numpy'][0] and initial['rng_numpy'][2:] == state['rng_numpy'][2:]
    assert torch.equal(initial['rng_torch'], state['rng_torch'])
    assert all(torch.equal(a, b) for a, b in zip(initial['rng_cuda'], state['rng_cuda']))
    confidence = state['confidence']; assert confidence_hash(confidence) == state['confidence_sha256']
    with np.load(RUNS/'geometric_confidence.npz') as geometry:
        for name, value in confidence.items():
            assert np.isfinite(value).all() and (value >= 0).all() and (value <= geometry[name]+1e-6).all()
            assert not value[geometry[name] == 0].any()
    interval = 20 if smoke else read(REPORT/'protocol.json')['confidence_interval']
    expected_rounds = [] if arm == 'fixed' else list(range(interval, n+1, interval))
    assert [event['step'] for event in state['confidence_history']] == expected_rounds
    if arm == 'fixed': assert confidence_hash(confidence) == confidence_hash(initial['confidence'])
    else: assert state['confidence_history'][-1]['confidence_sha256'] == confidence_hash(confidence)
    if smoke and n == 80:
        before = read(REPORT/f'smoke40_{arm}.json')
        assert before['passed'] and before['sampler'] == sequence[:40]
        assert before['checkpoint_sha256'] != sha(path)
        if arm == 'progressive':
            assert before['confidence_sha256'] == state['confidence_history'][1]['confidence_sha256']
    result = {'passed': True, 'arm': arm, 'iteration': n, 'checkpoint_sha256': sha(path),
        'optimizer_steps': counters, 'sampler': sequence if smoke and n == 40 else None,
        'confidence_sha256': confidence_hash(confidence), 'confidence_rounds': expected_rounds,
        'full_Adam_RNG_sampler_and_confidence_checked': True, 'geometric_upper_bound_checked': True,
        'nonzero_supervision_steps': sum(r['supervised_pixels'] > 0 for r in actual),
        'source_and_historical_checkpoints_preserved': True}
    write(REPORT/(f'smoke40_{arm}.json' if smoke and n == 40 else f'smoke80_{arm}.json' if smoke else f'formal_{arm}_audit.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--step', type=int, default=12000)
    parser.add_argument('--arm', choices=['fixed', 'progressive'])
    args = parser.parse_args()
    if args.freeze: freeze(); print('CROSS_TEMPORAL_FROZEN', flush=True); return
    verify(); result = audit(args.arm, args.smoke, args.step)
    print('CROSS_TEMPORAL_AUDITED', args.arm, args.step, result['passed'], flush=True)
    if args.smoke and args.step == 80:
        other = REPORT/f'smoke80_{"progressive" if args.arm == "fixed" else "fixed"}.json'
        if other.exists():
            write(REPORT/'smoke_verification.json', {'passed': True, 'independent_process40_to80_both_arms': True,
                'confidence_state_including_intermediate_refresh_restored': True, 'source_manifest_sha256': sha(REPORT/'source_freeze.json')})


if __name__ == '__main__': main()
