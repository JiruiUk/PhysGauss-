"""Isolate official pair prediction from unused flow/trajectory imports."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'experiments/staged_reconstruction/monst3r_geometry'
SOURCE = ROOT / 'third_party/monst3r_574cc77'
VENDOR = ROOT / 'third_party/monst3r_pair_vendor'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = REPORT / 'pair_vendor.json'
    assert not receipt.exists() and not VENDOR.exists()
    lock = json.loads((REPORT / 'asset_source.json').read_text())
    for relative, expected in lock['source_files_sha256'].items():
        assert sha(SOURCE / relative) == expected, relative
    for relative in ['dust3r', 'croco/models']:
        shutil.copytree(SOURCE / relative, VENDOR / relative)
    changes = {}
    path = VENDOR / 'dust3r/model.py'
    original = path.read_text(encoding='utf-8')
    unused = 'from third_party.raft import load_RAFT\n'
    assert original.count('load_RAFT') == 1 and original.count(unused) == 1
    path.write_text(original.replace(unused, ''), encoding='utf-8')
    changes['dust3r/model.py'] = 'Remove unused load_RAFT import; forward/model construction untouched.'
    path = VENDOR / 'dust3r/utils/misc.py'
    original = path.read_text(encoding='utf-8')
    eager = 'from dust3r.utils.vo_eval import save_trajectory_tum_format\n'
    assert original.count(eager) == 1
    updated = original.replace(eager, '').replace('def save_tum_poses(traj, path):\n', 'def save_tum_poses(traj, path):\n    ' + eager)
    path.write_text(updated, encoding='utf-8')
    changes['dust3r/utils/misc.py'] = 'Import trajectory evaluator lazily inside its only caller; no pair forward change.'
    result = {'source_commit': lock['source_commit'], 'croco_commit': lock['croco_commit'],
              'source_manifest_sha256': sha(REPORT / 'asset_source.json'),
              'vendor_root': VENDOR.relative_to(ROOT).as_posix(), 'changes': changes,
              'files_sha256': {p.relative_to(VENDOR).as_posix(): sha(p) for p in sorted(VENDOR.rglob('*')) if p.is_file()},
              'forward_math_or_weights_modified': False,
              'no_flow_SAM_or_global_alignment_reproduction_claim': True}
    receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('MONST3R_PAIR_VENDOR_READY', changes, flush=True)


if __name__ == '__main__':
    main()
