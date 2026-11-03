import subprocess
import sys
import pytest
from variants.staged_reconstruction import continuation


def test_continuation_disabled_does_not_import_cuda_renderer():
    p=subprocess.run([sys.executable,'-m','variants.staged_reconstruction.continuation','--stage','S4'],capture_output=True,text=True)
    assert p.returncode and 'opt-in' in p.stderr and 'ninja' not in p.stdout


def test_unreviewed_previous_stage_blocks_continuation(tmp_path,monkeypatch):
    monkeypatch.setattr(continuation,'REPORT',tmp_path)
    with pytest.raises(RuntimeError,match='reviewed admission'):continuation.parent_gate('S4')


def test_disabled_interior_does_not_read_scene_or_eligibility():
    assert continuation.interior_loss(None,None,None,None) is None
