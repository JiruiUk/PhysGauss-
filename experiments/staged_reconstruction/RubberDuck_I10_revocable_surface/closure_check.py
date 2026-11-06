"""Read-only final manifest and task Git-byte check; no CUDA or model updates."""
import hashlib
import json
from pathlib import Path
import subprocess

from variants.staged_reconstruction.i10_tracking.io import ROOT, write


def main():
    b = Path(__file__).parent
    v = b / 'v1'
    r = ROOT / 'runs/pidg_staged_reconstruction/RubberDuck_I10_revocable_surface/v1'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    f = read(v / 'final_review.json')
    for name, digest in f['artifacts_sha256'].items():
        assert sha(v / name) == digest, name
    manual = read(v / 'manual_review.json')
    assert len(manual['reviewed_images_sha256']) == 32
    for path, digest in manual['reviewed_images_sha256'].items():
        assert sha(ROOT / path) == digest, path
    for name in ['final_status.json', 'active.json', 'pipeline.json']:
        q = read(r / name)
        assert q['status'] == 'completed_not_promoted' and not q['pending_review']
    for arm in ['fixed_surface', 'revocable_surface']:
        q = read(r / 'formal' / arm / 'final_status.json')
        assert not q['pending_review'] and q['iteration'] == 2000
        assert q['checkpoint_sha256'] == sha(r / 'formal' / arm / 'checkpoint.pt')
    count = 0
    prefixes = ('variants/staged_reconstruction/i10_revocable_surface/',
                'experiments/staged_reconstruction/RubberDuck_I10_revocable_surface/',
                'tests/test_i10_revocable_surface.py')
    for path, digest in read(v / 'source_freeze.json').items():
        if path.startswith(prefixes):
            raw = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=ROOT)
            assert hashlib.sha256(raw).hexdigest() == digest, path
            assert sha(ROOT / path) == digest, path
            count += 1
    write(v / 'closure_validation.json', dict(
        passed=True, frozen_task_Git_and_worktree_bytes_checked=count,
        final_manifest_verified=len(f['artifacts_sha256']), actual_reviewed_images_verified=32,
        two_full_checkpoint_SHA_verified=True, closed_states_verified=True,
        no_training_or_decisions_modified=True,
        final_review_sha256=sha(v / 'final_review.json'),
        final_validation_sha256=sha(v / 'final_validation.json'),
        source_sha256=sha(Path(__file__))))
    print('Final manifest, 32 reviewed images, full checkpoints and closed states passed;', count, 'frozen task Git/source entries exact')


if __name__ == '__main__':
    main()
