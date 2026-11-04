"""Small checks for the new switch, gradient routes and sampler continuation."""
from types import SimpleNamespace
import random
import torch
from torch import nn
from variants.staged_reconstruction.rubber_joint.core import configure
from variants.staged_reconstruction.joint_depth_trust.core import self_depth_loss
from variants.staged_reconstruction.rubber_joint.runtime import next_index

def test_disabled_identity_no_mutation():
    scene=SimpleNamespace(static=nn.Linear(3,2).requires_grad_(False))
    assert configure(scene) is scene
    assert not any(p.requires_grad for p in scene.static.parameters())

def test_only_appearance_open_teacher_excluded():
    static=nn.Module()
    for name in ['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation']:
        static.register_parameter(name,nn.Parameter(torch.ones(2),requires_grad=False))
    scene=SimpleNamespace(static=static);before=set(static.state_dict())
    teacher=configure(scene,True)
    assert teacher is not static and not any(p.requires_grad for p in teacher.parameters())
    assert set(static.state_dict())==before
    assert {n for n,p in static.named_parameters() if p.requires_grad}=={'_opacity','_features_dc','_features_rest'}

def test_self_reference_no_teacher_gradient():
    D=torch.tensor([3.,4.],requires_grad=True);a=torch.tensor([.8,.9],requires_grad=True)
    teacher_D=torch.tensor([2.,3.],requires_grad=True);teacher_a=torch.ones(2,requires_grad=True)
    self_depth_loss(D,a,teacher_D,teacher_a).backward()
    assert D.grad.isfinite().all() and a.grad.isfinite().all()
    assert teacher_D.grad is None and teacher_a.grad is None

def test_sampler_exact_inherited_then_refill():
    state=random.getstate()
    try:
        random.seed(42);stack=[8,3,5];rng=random.getstate()
        assert [next_index(stack) for _ in range(3)]==[5,3,8]
        assert random.getstate()==rng
        observed=[next_index(stack) for _ in range(120)]
        assert sorted(observed)==list(range(120))
    finally:random.setstate(state)
