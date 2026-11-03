import numpy as np
import torch
from variants.staged_reconstruction.stage2.identity import advance, reciprocal, directional


def test_variable_count_reciprocal_rejects_merge_and_split():
    assert reciprocal(np.array([0,1,-1]), np.array([0,1])).tolist() == [0,1,-1]
    assert reciprocal(np.array([0,0]), np.array([0])).tolist() == [-1,-1]
    assert reciprocal(np.array([0]), np.array([0,0])).tolist() == [-1]


def test_lost_identity_cannot_revive_from_later_component_number():
    lost = advance(np.array([0,1,2]), np.array([0,-1,1]))
    assert advance(lost, np.array([1,0,2])).tolist() == [1,-1,0]


def test_zero_flow_preserves_separate_components_and_merge_is_unknown():
    source = np.full((32,64),-1,np.int64); source[4:28,3:25]=0; source[4:28,38:60]=1
    flow=torch.zeros(32,64,2)
    assert directional(source,source,flow,flow).tolist()==[0,1]
    target=np.where(source>=0,0,-1)
    assert directional(source,target,flow,flow).tolist()==[-1,-1]


def test_tiny_advection_support_not_a_valid_identity():
    source=np.full((32,32),-1,np.int64);source[5:8,5:8]=0
    flow=torch.zeros(32,32,2)
    assert directional(source,source,flow,flow).tolist()==[-1]


def test_compose_keeps_static_constant_and_dynamic_gradients():
    from types import SimpleNamespace
    from variants.staged_reconstruction.stage2.model import compose
    names=('get_xyz','get_scaling','get_rotation','get_opacity','get_features')
    static=SimpleNamespace(**{n:torch.ones(2,3) for n in names})
    dynamic=SimpleNamespace(**{n:torch.full((4,3),2.,requires_grad=True) for n in names})
    mixed=compose(static,[dynamic]);mixed.get_xyz.sum().backward()
    assert mixed.get_xyz.shape==(6,3) and torch.equal(mixed.get_xyz[:2],static.get_xyz)
    assert torch.equal(dynamic.get_xyz.grad,torch.ones_like(dynamic.get_xyz))
    assert not static.get_xyz.requires_grad


def test_disabled_object_stage_does_not_load_renderer():
    import subprocess,sys
    p=subprocess.run([sys.executable,'-m','variants.staged_reconstruction.stage2.train'],capture_output=True,text=True)
    assert p.returncode!=0 and 'opt-in' in p.stderr and 'ninja' not in p.stdout
