import os
from pathlib import Path
import subprocess
import sys
import pytest
from variants.staged_reconstruction.surface_trajectory.admission import evidence_gates


def test_numerical_split_alone_does_not_admit_highlight_tracks_for_training():
    bindings = [dict(time_band=i % 3) for i in range(60)]
    split = dict(admitted=True, training_track_ids=list(range(30)), holdout_track_ids=list(range(30, 60)),
                 holdout_groups=list(range(20)))
    result = evidence_gates(bindings, split, dict(actual_visual_inspection_completed=True, surface_feature_evidence_sufficient=False))
    assert not result['evidence_admitted'] and result['failed_gates'] == ['stable_surface_feature_evidence_sufficient']
    assert not result['formal_training_admitted']


def test_twenty_three_tracks_with_twelve_groups_cannot_be_promoted():
    bindings = [dict(time_band=0)]*12+[dict(time_band=1)]*3+[dict(time_band=2)]*8
    result = evidence_gates(bindings, dict(admitted=False),
        dict(actual_visual_inspection_completed=True, surface_feature_evidence_sufficient=False))
    assert not result['evidence_admitted'] and not result['formal_training_admitted']
    assert result['counts_by_time_band'] == [12, 3, 8]


@pytest.mark.parametrize('module', ['prepare', 'bind', 'review', 'diagnostics'])
def test_disabled_entry_does_not_require_inputs_or_cuda_setup(tmp_path, module):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    process = subprocess.run([sys.executable, '-m', 'variants.staged_reconstruction.surface_trajectory.'+module],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert process.returncode != 0 and 'opt-in' in process.stderr
    assert 'ninja' not in process.stdout and 'Traceback' not in process.stderr
