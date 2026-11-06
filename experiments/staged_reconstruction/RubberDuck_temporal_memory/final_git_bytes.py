"""Repair only newline-normalized evidence blobs; preserve worktree/results."""
from pathlib import Path
import hashlib
import subprocess
from variants.staged_reconstruction.i10_tracking.io import ROOT, read, write, sha


def main():
    base = Path(__file__).parent / 'v1'
    final = read(base / 'final_review.json')
    mismatches = []
    for name, digest in final['artifacts_sha256'].items():
        path = (base / name).relative_to(ROOT).as_posix()
        original = (ROOT / path).read_bytes()
        assert hashlib.sha256(original).hexdigest() == digest, path
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != digest:
            assert original.replace(b'\r\n', b'\n') == committed, path
            mismatches.append(dict(path=path, expected_sha256=digest,
                                   committed_sha256=hashlib.sha256(committed).hexdigest()))
    assert {Path(q['path']).name for q in mismatches} == {'unit_tests.json', 'independent_input_audit.json'}
    write(base / 'git_final_manifest_attempt1.json', dict(
        passed=False, mismatches=mismatches,
        reason='Two earlier evidence JSON blobs were LF-normalized before task raw-byte attributes existed; training and worktree hashes unchanged',
        training_or_worktree_modified=False))
    for item in mismatches:
        raw = (ROOT / item['path']).read_bytes()
        blob = subprocess.check_output(['git', 'hash-object', '-w', '--stdin'], input=raw, cwd=ROOT).decode().strip()
        subprocess.run(['git', 'update-index', '--cacheinfo', '100644,' + blob + ',' + item['path']], cwd=ROOT, check=True)
        staged = subprocess.check_output(['git', 'show', ':' + item['path']], cwd=ROOT)
        assert hashlib.sha256(staged).hexdigest() == item['expected_sha256']
    write(base / 'git_final_storage_repair.json', dict(
        passed=True, exact_raw_bytes_staged=mismatches,
        training_or_worktree_modified=False, source_sha256=sha(Path(__file__))))
    print('FINAL_EVIDENCE_RAW_BYTES_STAGED', len(mismatches))


if __name__ == '__main__':
    main()
