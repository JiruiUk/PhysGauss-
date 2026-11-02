from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'experiments/association_training'
DATA = ROOT / 'runs/pidg_association_training/training_data.npz'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare():
    old = json.loads((ROOT / 'experiments/object_membership/results.json').read_text())
    proof = json.loads((ROOT / 'experiments/object_membership/verification.json').read_text())
    assert proof['passed']
    path = ROOT / 'output/object_membership/membership.npz'
    assert sha(path) == old['arrays_sha256']
    with np.load(path) as z:
        labels = z['early_2_27_historical_owner'][-1].copy()
        frames = z['early_2_27_frames'].copy()
        mapping = z['early_2_27_mapping'].copy()
    with np.load(ROOT / 'output/surface_support/image_tracks.npz') as z:
        components = {int(n): c for n, c in zip(z['frames'], z['components'])}
    masks = []
    reference = []
    for t, n in enumerate(frames):
        mm = np.stack([components[n] == c if c >= 0 else np.zeros((225, 400), bool) for c in mapping[t]])
        with np.load(ROOT / f'runs/pidg_short_clip_diagnostic/cache/native{n:04d}.npz') as z:
            mass = z['mass']
        ref = [float(mass[labels == k, c].sum(dtype=np.float64) / max(mm[k].sum(), 1)) if c >= 0 else 0 for k, c in enumerate(mapping[t])]
        masks.append(mm)
        reference.append(ref)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        np.savez_compressed(DATA, labels=labels, anchor_frames=frames, anchor_masks=np.asarray(masks),
                            anchor_reference=np.asarray(reference, dtype=np.float32), source_checkpoint_sha256=np.asarray(old['checkpoint_sha256']))
    else:
        with np.load(DATA) as z:
            assert np.array_equal(z['labels'], labels) and np.array_equal(z['anchor_masks'], masks)
            assert np.array_equal(z['anchor_reference'], np.asarray(reference, dtype=np.float32))
    result = {'path': str(DATA), 'sha256': sha(DATA), 'source_checkpoint_sha256': old['checkpoint_sha256'],
              'point_count': len(labels), 'labelled_points': int((labels >= 0).sum()),
              'per_object_counts': [int((labels == k).sum()) for k in range(4)], 'anchor_frames': frames.tolist(),
              'late_labels_used': False, 'physical_material_IDs': False}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'training_data.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
