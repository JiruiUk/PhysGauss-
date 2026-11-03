import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'experiments/staged_reconstruction/S1_joint_rgb/accepted_recovery'


def test_all_original_numeric_training_values_and_quality_gates_preserved():
    old = json.loads((BASE.parent/'protocol.json').read_text())
    new = json.loads((BASE/'protocol.json').read_text())
    for key, value in old.items():
        if key != 'stage': assert new[key] == value
    assert new['Adam_eps'] == 1e-15 and new['human_approved_recovery_acceptance']
    assert new['original_full40_gate_failed_and_retained']


def test_all_launcher_phases_disabled_before_any_native_or_write():
    for phase in ['train', 'pipeline', 'qualify', 'prepare', 'qualification-worker', 'evaluate']:
        result = subprocess.run([sys.executable, str(BASE/'launch.py'), '--phase', phase], capture_output=True, text=True)
        assert result.returncode != 0 and 'Accepted J1 experiment is disabled' in result.stderr
