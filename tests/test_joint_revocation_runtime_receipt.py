import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('j0_receipt_repair', ROOT/'experiments/staged_reconstruction/S1_joint_revocation_audit/run_worker_receipt_repair.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_generated_nonexistent_torch_source_and_unfiled_modules_are_skipped():
    assert MODULE.source_files([SimpleNamespace(__file__='_nonexistent_torch_virtual_ops.py'), SimpleNamespace()]) == {}


def test_real_workspace_source_remains_hash_bound():
    files = MODULE.source_files([SimpleNamespace(__file__=__file__)])
    key = Path(__file__).relative_to(ROOT).as_posix()
    assert list(files) == [key] and len(files[key]) == 64
