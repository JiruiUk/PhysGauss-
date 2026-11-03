"""Independent S1-A state/sampler/source audit; no historical weight mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.staged_reconstruction.prepare_data import read, write, sha
from variants.staged_reconstruction.automatic_static.discover import REPORT, RUNS
from variants.staged_reconstruction.automatic_static.initialize import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['capture40', 'verify80', 'formal'], required=True)
    args = parser.parse_args()
    verify()
    directory = RUNS/('formal' if args.mode == 'formal' else 'smoke')
    path = directory/'checkpoint.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    initial = torch.load(directory/'initial.pt', map_location='cpu', weights_only=False)
    n = 12000 if args.mode == 'formal' else 40 if args.mode == 'capture40' else 80
    assert state['iteration'] == n and read(directory/'checkpoint.json')['sha256'] == sha(path)
    assert state['protocol_sha256'] == initial['protocol_sha256'] == sha(REPORT/'protocol.json')
    assert state['static_only'] and state['dynamic_parameters'] == 0
    assert all(v.isfinite().all() for v in state['model'].values())
    assert all(v.isfinite().all() for s in state['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v))
    counters = [int(v['step']) for v in state['optimizer']['state'].values()]
    assert len(counters) == 6 and all(v == n for v in counters)
    count = len(read(directory/'sampling_contract.json')['frames'])
    rng = random.Random(); rng.setstate(initial['rng_python'])
    stack, sequence = [], []
    for _ in range(n):
        if not stack:
            stack = rng.sample(range(count), count)
        sequence.append(stack.pop())
    actual = [json.loads(line) for line in (directory/'sampling.jsonl').read_text().splitlines()]
    assert [r['step'] for r in actual] == list(range(1, n+1))
    assert [r['frame_index'] for r in actual] == sequence
    assert rng.getstate() == state['rng_python'] and stack == state['camera_stack']
    assert np.array_equal(initial['rng_numpy'][1], state['rng_numpy'][1])
    assert initial['rng_numpy'][0] == state['rng_numpy'][0] and initial['rng_numpy'][2:] == state['rng_numpy'][2:]
    assert torch.equal(initial['rng_torch'], state['rng_torch'])
    assert len(initial['rng_cuda']) == len(state['rng_cuda'])
    assert all(torch.equal(a, b) for a, b in zip(initial['rng_cuda'], state['rng_cuda']))
    if args.mode == 'capture40':
        write(REPORT/'smoke40.json', {'passed': True, 'checkpoint_sha256': sha(path),
            'sampler': sequence, 'parameter_sha256': hashlib.sha256(b''.join(v.numpy().tobytes() for v in state['model'].values())).hexdigest()})
        print('AUTOMATIC_STATIC_SMOKE40_CAPTURED', flush=True)
        return
    if args.mode == 'verify80':
        before = read(REPORT/'smoke40.json')
        assert before['passed'] and before['sampler'] == sequence[:40]
        assert before['checkpoint_sha256'] != sha(path)
    else:
        assert read(REPORT/'smoke_verification.json')['passed']
    result = {'passed': True, 'iteration': n, 'checkpoint_sha256': sha(path),
        'full_state_Adam_and_all_RNG_and_sampler_verified': True, 'optimizer_state_count': len(counters),
        'all_source_input_protocol_hashes_match': True, 'static_only': True,
        'independent_process_40_to80_resume': args.mode == 'verify80',
        'native_renderer_exact_parity': read(REPORT/'native_parity.json')['passed'],
        'physical_geometry_or_complete_staticity_certified': False,
        'sampling_frames': count, 'static_points': read(REPORT/'initialization.json')['static_points']}
    write(REPORT/('formal_state_audit.json' if args.mode == 'formal' else 'smoke_verification.json'), result)
    print('AUTOMATIC_STATIC_STATE_VERIFIED', n, counters, flush=True)


if __name__ == '__main__':
    main()
