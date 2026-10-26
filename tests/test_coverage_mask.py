import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from variants.coverage_tuning.model import mask_loss, coverage_metrics
from variants.coverage_tuning.data import audit_training, read_alpha, frame_paths


def test_mask_is_exact_l1_with_correct_gradient_and_detached_gt():
    a = torch.tensor([[[.1,.9],[.3,.6]]], requires_grad=True)
    gt = torch.tensor([[[0.,1.],[1.,0.]]], requires_grad=True)
    loss = mask_loss(a, gt)
    assert loss.item() == pytest.approx(.375)
    loss.backward()
    assert torch.equal(a.grad, torch.tensor([[[.25,-.25],[-.25,.25]]]))
    assert gt.grad is None
    with pytest.raises(ValueError): mask_loss(a, gt[0])


def test_hole_metric_ignores_background_and_boundary():
    gt = torch.zeros(1,15,15); gt[:,2:13,2:13] = 1
    a = gt.clone(); a[:,7,7] = 0
    m = coverage_metrics(a,gt)
    assert m['interior_pixels'] == 25
    assert m['interior_hole_rate'] == pytest.approx(1/25)
    assert m['mask_iou'] == pytest.approx(120/121)


def test_audit_uses_only_train_alpha_and_refuses_rgb(tmp_path):
    a = np.zeros((20,30,4), dtype=np.uint8); a[5:15,8:20,3] = 255
    Image.fromarray(a).save(tmp_path/'0002.png')
    for split, name in [('train','0002'),('test','0004')]:
        (tmp_path/f'transforms_{split}.json').write_text(json.dumps({'frames':[{'file_path':name}]}))
    # Validation PNG intentionally absent: auditing training may not open it.
    assert audit_training(tmp_path)['count'] == 1
    assert read_alpha(tmp_path/'0002.png',(15,10)).shape == (1,10,15)
    Image.fromarray(a[:,:,:3]).save(tmp_path/'0002.png')
    with pytest.raises(ValueError): audit_training(tmp_path)


def test_runtime_flags_off_never_accesses_targets_or_rng(tmp_path):
    from variants.coverage_tuning.runtime import Tuning
    config = tmp_path/'config.json'; config.write_text(json.dumps({'enabled':False}))
    runtime = Tuning(config,tmp_path)
    before = torch.get_rng_state().clone()
    assert runtime.coverage_loss(1,None,None,None,None) is None
    runtime.after_update(1,None)
    assert torch.equal(before,torch.get_rng_state())


def test_non_train_view_cannot_enter_mask_loss(tmp_path):
    from variants.coverage_tuning.runtime import Tuning
    config = tmp_path/'config.json'; config.write_text(json.dumps({'enabled':True,'mask_weight':.1}))
    runtime = Tuning(config,tmp_path)
    runtime.train_names = {'0002'}
    with pytest.raises(ValueError, match='training'):
        runtime.coverage_loss(1,SimpleNamespace(image_name='0001'),None,None,None)
