"""RGB-only official Small inference. No GT import or Gaussian optimization."""
import argparse
import msvcrt
import os
import platform
from pathlib import Path
import numpy as np
from PIL import Image
from ..prepare_data import ROOT, read, write, sha
from .prepare import BASE, REPORT, RUNS, PARENT, SOURCE, verify, stopped
from .model import load_model, predict
from .core import calibrate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-pretrained-depth', action='store_true')
    args = parser.parse_args()
    assert args.enable_pretrained_depth, 'Pretrained geometry is opt-in'
    assert not stopped(), 'Saved user stop; never auto-resume'
    verify(); assert not (REPORT / 'inference.json').exists(), 'Preserve completed prior'
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = (RUNS / 'inference.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    model = None
    try:
        p = read(BASE / 'protocol.json')
        assert load_model(None, None) is None
        model = load_model(ROOT, p, True)
        model_parameters = sum(t.numel() for t in model.parameters())
        import torch
        results = {}
        for case in p['synthetic_cases']:
            train = ROOT / f'data/derived/clean_static_synthetic/v2/{case}/train'
            meta = read(train / 'cameras.json')
            directory = RUNS / case; directory.mkdir(exist_ok=True)
            near, far = read(SOURCE / f'{case}_inference.json')['camera_derived_depth_range']
            rows = []
            for i, row in enumerate(meta['frames']):
                assert not stopped(), 'Saved user stop; retain view files and release GPU'
                path = directory / f'view_{i:03d}.npz'
                calibration_path = directory / f'view_{i:03d}.json'
                rgb_path = train / row['image']
                anchor_path = PARENT / case / f'view_{i:03d}.npz'
                if path.exists() and calibration_path.exists():
                    info = read(calibration_path)
                    assert info['output_sha256'] == sha(path)
                    assert info['rgb_sha256'] == sha(rgb_path) and info['anchor_sha256'] == sha(anchor_path)
                else:
                    assert not path.exists(), 'Incomplete view: preserve and inspect before resuming'
                    with Image.open(rgb_path) as image:
                        assert image.mode == 'RGB'
                        rgb = np.asarray(image)
                    original = object()
                    assert predict(original) is original and calibrate(original)[0] is original
                    raw = predict(None, rgb, model, p['official_input_size'], True)
                    with np.load(anchor_path) as anchor:
                        tight = anchor['tight']
                        uv = anchor['uv'][tight].copy(); z = anchor['depth'][tight].copy()
                    aligned, info = calibrate(None, raw, uv, z, near, far, p, True)
                    fit_ids = info.pop('fit_ids'); held_ids = info.pop('held_ids')
                    np.savez_compressed(path, raw_relative=raw, aligned_depth=aligned,
                                        anchor_uv=uv, anchor_depth=z, fit_ids=fit_ids, held_ids=held_ids)
                    info.update({'image': row['image'], 'time': row['time'], 'rgb_sha256': sha(rgb_path),
                                 'anchor_sha256': sha(anchor_path), 'output_sha256': sha(path)})
                    write(calibration_path, info)
                rows.append(info)
                write(RUNS / 'state.json', {'status': 'inference', 'pid': os.getpid(), 'case': case,
                      'completed': i+1, 'total': len(meta['frames'])})
                if i % 12 == 0:
                    print('PRETRAINED_DEPTH', case, i+1, 'calibrated', sum(r['calibrated'] for r in rows), flush=True)
            results[case] = {'views': len(rows), 'calibrated_views': sum(r['calibrated'] for r in rows), 'frames': rows}
        del model; model = None; torch.cuda.empty_cache()
        verify()
        write(REPORT / 'inference.json', {'GT_read': False, 'Gaussian_training': False, 'results': results,
              'source_manifest_sha256': sha(REPORT / 'source_freeze.json'),
              'model_parameters': model_parameters,
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                              'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)},
              'outputs': {q.relative_to(ROOT).as_posix(): sha(q) for q in RUNS.rglob('view_*') if q.is_file()}})
        write(RUNS / 'state.json', {'status': 'inferred_pending_independent_evaluation'})
        print('PRETRAINED_DEPTH_COMPLETE', {k: v['calibrated_views'] for k, v in results.items()}, flush=True)
    except Exception as error:
        write(RUNS / 'state.json', {'status': 'saved_stopped' if stopped() else 'failed', 'error': str(error), 'pid': os.getpid()})
        raise
    finally:
        if model is not None:
            del model
            import torch
            torch.cuda.empty_cache()
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__':
    main()
