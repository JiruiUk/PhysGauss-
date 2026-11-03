import numpy as np
import torch
from variants.staged_reconstruction.model import observation_masks, masked_objective, eligible_voxels


def test_masks_preserve_unknown_and_occlusion():
    rgba = np.zeros((20, 20, 4), np.float32)
    rgba[2:6, 2:6] = [.1, .25, .8, 1]
    rgba[10:14, 10:14] = [.9, .8, .1, 1]
    rgba[16:18, 2:4] = [.7, .7, .7, 1]
    positive, negative = observation_masks(rgba, dilation=1)
    assert positive[3, 3] and not negative[3, 3]
    assert not positive[11, 11] and not negative[11, 11]
    assert not negative[9, 11] and not negative[16, 2]
    assert negative[0, 19]


def test_unknown_pixels_have_zero_gradient():
    rgb = torch.full((3, 3, 3), .5, requires_grad=True)
    alpha = torch.full((1, 3, 3), .5, requires_grad=True)
    positive = torch.zeros(3, 3, dtype=torch.bool); positive[0, 0] = True
    negative = torch.zeros_like(positive); negative[2, 2] = True
    loss, _ = masked_objective(rgb, alpha, torch.zeros_like(rgb), positive, negative)
    gr, ga = torch.autograd.grad(loss, [rgb, alpha])
    assert torch.equal(gr[:, 1, 1], torch.zeros(3)) and ga[0, 1, 1] == 0
    assert gr[:, 0, 0].abs().sum() > 0 and ga[0, 0, 0] < 0 and ga[0, 2, 2] > 0


def test_no_positive_supervision_fails_explicitly():
    import pytest
    with pytest.raises(ValueError):
        masked_objective(torch.zeros(3,2,2), torch.zeros(1,2,2), torch.zeros(3,2,2),
                         torch.zeros(2,2,dtype=torch.bool), torch.ones(2,2,dtype=torch.bool))


def test_carving_requires_multitime_support_and_rejects_background():
    blue = np.array([20,20,20,2]); background = np.array([0,8,0,0]); seen = np.array([30]*4)
    thirds = np.array([[4,5,4],[5,5,5],[20,0,0],[1,0,1]])
    assert eligible_voxels(blue, background, seen, thirds).tolist() == [True,False,False,False]


def test_static_bank_has_no_dynamic_or_material_parameters_and_is_bounded(tmp_path):
    from variants.staged_reconstruction.static_gaussian import StaticGaussian
    path=tmp_path/'initial.npz'
    np.savez(path,xyz=np.zeros((2,3),np.float32),rgb=np.full((2,3),.3,np.float32),spacing=.04)
    bank=StaticGaussian(path,device='cpu')
    assert set(dict(bank.named_parameters()))=={'_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation'}
    with torch.no_grad():bank._xyz.fill_(100);bank._scaling.fill_(100)
    bank.constrain()
    assert float(bank.get_xyz.abs().max()) <= .080001
    assert float(bank.get_scaling.max()) <= .100001


def test_disabled_command_never_initializes_renderer():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-m','variants.staged_reconstruction.train_static'],capture_output=True,text=True)
    assert result.returncode!=0 and 'opt-in' in result.stderr
    assert 'ninja' not in result.stdout
