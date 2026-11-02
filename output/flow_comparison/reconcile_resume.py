"""Archive observations beyond a saved boundary before an interrupted resume.

Run only after confirming the old coordinator and worker have exited. Never
changes weights, configuration, checkpoint selection, or frozen training code.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/pidg_flow_comparison/K1_flow_independent/rubber_duck'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
metadata = read(RUN/'checkpoints/resume.json')
boundary = metadata['iteration']
assert boundary == 41000
assert sha(RUN/'checkpoints/resume.pt') == metadata['sha256']
stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
archive = ROOT/'output/flow_comparison/resume_archives'/stamp
archive.mkdir(parents=True, exist_ok=False)
changes = []
for path in RUN.glob('*.jsonl'):
    lines = path.read_bytes().splitlines(keepends=True)
    kept = [line for line in lines if json.loads(line).get('iteration', -1) <= boundary]
    if len(kept) == len(lines):
        continue
    before = sha(path)
    shutil.copy2(path, archive/path.name)
    assert sha(archive/path.name) == before
    temporary = path.with_suffix('.reconcile.tmp')
    temporary.write_bytes(b''.join(kept))
    temporary.replace(path)
    changes.append({'file': str(path.relative_to(ROOT)), 'original_sha256': before,
                    'retained_sha256': sha(path), 'retained_rows': len(kept),
                    'archived_unsaved_rows': len(lines)-len(kept)})
receipt = {'reason': 'Previous coordinator and worker absent; last log 41760, saved state 41000.',
           'created_utc': datetime.now(timezone.utc).isoformat(), 'boundary': boundary,
           'checkpoint_sha256': metadata['sha256'], 'archive': str(archive), 'files': changes,
           'weights_or_frozen_source_changed': False,
           'limitation': 'Local momentum KDTree cache is rebuilt on resume, not serialized; full optimizer/RNG restored, but bitwise uninterrupted equivalence is not claimed.'}
(ROOT/'experiments/flow_comparison/resume_reconciliation.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt, indent=2))
