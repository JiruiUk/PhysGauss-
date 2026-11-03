"""Count-free population and recovery contracts, before full training."""
import subprocess
import sys

import numpy as np
import pytest
import torch

from variants.staged_reconstruction.union.core import dynamic_mask, initialize_union, union_coverage


@pytest.mark.parametrize('regions', [1, 3, 7])
def test_global_budget_does_not_depend_on_number_of_disconnected_regions(regions):
    rgba = np.zeros((32, 64, 4), np.float32)
    for k in range(regions):
        rgba[4:10, 2+k*8:6+k*8] = [1., .8, .1, 1.]
    rgba[20:28, 10:50] = [.1, .2, 1., 1.]
    camera = np.eye(4)
    intrinsic = np.array([[60., 0, 32], [0, 60., 16], [0, 0, 1]])
    static = np.array([[x, y, 4.] for x in np.linspace(-2, 2, 12) for y in [-.5, .5]])
    result = initialize_union(rgba, camera, intrinsic, static, 113, 17)
    assert result['xyz'].shape == (113, 3)
    assert result['rgb'].shape == (113, 3)
    assert np.isfinite(result['xyz']).all()
    assert not any('id' in key or 'count' in key for key in result)
    assert dynamic_mask(rgba).sum() == regions*24
    duplicate = initialize_union(rgba, camera, intrinsic, static, 113, 17)
    assert np.array_equal(result['xyz'], duplicate['xyz'])


def test_missing_foreground_fails_instead_of_inventing_objects():
    with pytest.raises(ValueError, match='foreground'):
        initialize_union(np.zeros((8, 8, 4)), np.eye(4), np.eye(3), np.ones((70, 3)), 12, 1)


def test_union_coverage_is_permutation_invariant_and_has_finite_position_gradients():
    from types import SimpleNamespace
    camera = SimpleNamespace(full_proj_transform=torch.eye(4), image_width=64, image_height=32)
    positions = torch.tensor([[-.5, .2, 1.], [.4, -.1, 1.]], requires_grad=True)
    mask = torch.zeros(16, 32, dtype=torch.bool)
    mask[6:9, 3:7] = True
    mask[6:9, 23:27] = True
    loss = union_coverage(positions, camera, mask)
    assert torch.allclose(loss, union_coverage(positions.flip(0), camera, mask))
    loss.backward()
    assert positions.grad.isfinite().all() and positions.grad.abs().sum() > 0


def test_entry_is_disabled_without_explicit_flag():
    result = subprocess.run([sys.executable, '-m', 'variants.staged_reconstruction.union.train'], capture_output=True, text=True)
    assert result.returncode != 0 and 'opt-in' in result.stderr
    assert 'ninja' not in result.stdout


def test_count_free_source_does_not_load_four_group_initialization_or_ledger():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]/'variants/staged_reconstruction/union'
    forbidden = ['S2_observations', 'S2_object_', 'S2_identity', 'S2_data_receipt', 'range(4)', 'object_count', 'object_ids']
    for path in root.glob('*.py'):
        for token in forbidden:
            assert token not in path.read_text(encoding='utf8'), (path, token)
