"""Meaningful denominator, missingness, independent mixing and opt-in checks."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from experiments.staged_reconstruction.S1_joint_attribution.core import (
    labels, transitions, summarize, mixed_payload, attribution)


def test_transition_denominator_and_closure():
    a = np.array([[0, 0, 0], [1, 1, 2]], dtype=np.uint8)
    b = np.array([[0, 1, 2], [0, 1, 0]], dtype=np.uint8)
    r = transitions(a, b, np.ones_like(a, dtype=bool))
    assert r['denominator'] == 6 and r['lost_accurate'] == r['gained_accurate'] == 2
    assert r['lost_to_missing'] == r['lost_to_valid_inaccurate'] == 1
    assert r['net_accurate_change'] == 0
    assert summarize(np.asarray(r['matrix_old_rows_new_columns'])) == r


def test_missing_not_zero_error_and_original_boundaries():
    alpha = np.array([.5, .5001, 0., 1., 1.], dtype=np.float32)
    d = np.array([0., 0., 0., float('nan'), .0625], dtype=np.float32)
    gt = np.array([0., 0., 0., 1., .125], dtype=np.float32)
    cls, _ = labels(alpha, d, gt)
    assert cls.tolist() == [2, 0, 2, 2, 1]


def test_mix_clones_only_requested_fields_and_rejects_geometry_change():
    p = {k: torch.tensor([1.]) for k in ['_xyz', '_opacity', '_features_dc', '_features_rest']}
    c = {k: v.clone() if k == '_xyz' else v+1 for k, v in p.items()}
    result = mixed_payload(p, c, 'cp')
    assert result['_opacity'].item() == 2 and result['_features_dc'].item() == 1
    result['_xyz'].zero_(); assert p['_xyz'].item() == 1
    c['_xyz'].add_(1)
    with pytest.raises(AssertionError): mixed_payload(p, c, 'cc')


def test_symmetric_marginal_closes_with_interaction():
    r = attribution({'pp': 5., 'cp': 3., 'pc': 2., 'cc': 1.})
    assert r['opacity_symmetric_marginal_change'] == -1.5
    assert r['SH_symmetric_marginal_change'] == -2.5 and r['interaction'] == 1.


def test_default_off_does_not_import_torch_or_read_checkpoint():
    path = Path(__file__).resolve().parents[1]/'experiments/staged_reconstruction/S1_joint_attribution/audit.py'
    script = "import runpy,sys; sys.argv=[%r]; runpy.run_path(%r,run_name='__main__'); assert 'torch' not in sys.modules" % (str(path), str(path))
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'DISABLED' in result.stdout
