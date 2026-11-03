"""Independent process model reload and official preprocessing replay; no GT."""
import numpy as np
from PIL import Image
from ..prepare_data import ROOT, read, write, sha
from .prepare import BASE, REPORT, RUNS, verify, stopped
from .model import load_model, predict


def main():
    assert not stopped()
    verify()
    receipt = read(REPORT / 'inference.json')
    for name, h in receipt['outputs'].items():
        assert sha(ROOT / name) == h, name
    p = read(BASE / 'protocol.json')
    model = load_model(ROOT, p, True)
    checks = {}
    for case in p['synthetic_cases']:
        train = ROOT / f'data/derived/clean_static_synthetic/v2/{case}/train'
        meta = read(train / 'cameras.json')
        rgb = np.asarray(Image.open(train / meta['frames'][0]['image']))
        fresh = predict(None, rgb, model, p['official_input_size'], True)
        with np.load(RUNS / case / 'view_000.npz') as data:
            old = data['raw_relative']
            assert np.array_equal(old, fresh), 'Independent-process official inference differs'
        checks[case] = {'exact': True, 'maximum_absolute_difference': float(abs(old-fresh).max())}
    del model
    import torch
    torch.cuda.empty_cache()
    write(REPORT / 'replay.json', {'passed': True, 'independent_process_model_reload': True,
          'Gaussian_Adam_RNG_recovery_claim': False, 'checks': checks,
          'inference_sha256': sha(REPORT / 'inference.json')})
    print('PRETRAINED_DEPTH_INDEPENDENT_REPLAY_EXACT', checks, flush=True)


if __name__ == '__main__':
    main()
