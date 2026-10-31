from types import SimpleNamespace
import torch
import pytest
from variants.object_id.runtime import apply_warp,snapshot_save,snapshot_load
from variants.object_id.model import ObjectMotion


def test_disabled_hook_returns_same_tensor():
    warp=torch.randn(4,7)
    assert apply_warp(SimpleNamespace(object_model=None),None,None,warp) is warp


def test_next_frame_query_keeps_primary_image_regularization_state():
    model=ObjectMotion(4);x=torch.randn(30,3);model.initialize(x)
    holder=SimpleNamespace(object_model=model,object_cache=None,object_iteration=1,
        object_config={'object_start':0,'object_ramp':1})
    warp=torch.zeros(30,7);warp[:,0]=1
    apply_warp(holder,x,torch.zeros(30,1),warp)
    primary=holder.object_cache
    next_warp=warp.clone();next_warp[:,4:]=1
    apply_warp(holder,x,torch.ones(30,1),next_warp)
    assert holder.object_cache is primary


def test_selected_snapshot_restores_iteration_and_model(tmp_path):
    model=ObjectMotion(4);model.initialize(torch.randn(30,3))
    source=SimpleNamespace(object_model=model,object_iteration=123,scale_xyz=torch.ones(3),object_config={})
    snapshot_save(source,tmp_path)
    target=SimpleNamespace(object_model=ObjectMotion(4),scale_xyz=torch.ones(3),object_config={})
    snapshot_load(target,tmp_path,tmp_path,'best')
    assert target.object_iteration==123
    for k,v in model.state_dict().items():assert torch.equal(v,target.object_model.state_dict()[k])


def test_missing_trained_snapshot_fails_closed(tmp_path):
    holder=SimpleNamespace(object_model=ObjectMotion(4),scale_xyz=torch.ones(3),object_config={'object_start':30})
    with pytest.raises(ValueError,match='Missing trained object'):
        snapshot_load(holder,tmp_path,tmp_path,'100')
