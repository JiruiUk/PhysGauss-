from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from variants.staged_reconstruction.colorfree_joint.core import validate_payload, replace_static


def payload(n=3):
    return dict(_xyz=torch.zeros(n,3),_features_dc=torch.zeros(n,1,3),_features_rest=torch.zeros(n,15,3),
                _opacity=torch.zeros(n,1),_scaling=torch.zeros(n,3),_rotation=torch.zeros(n,4),
                initial_xyz=torch.zeros(n,3),point_ids=torch.arange(n),born_at=torch.zeros(n,dtype=torch.long))


def test_default_off_identity():
    scene=SimpleNamespace(static=object(),dynamic=object());old=scene.static
    assert replace_static(scene) is scene and scene.static is old


def test_only_static_replaced_and_source_unchanged():
    state=payload();before=deepcopy(state);bank=SimpleNamespace(state_dict=lambda:state)
    scene=SimpleNamespace(static=object(),dynamic=object());dynamic=scene.dynamic
    assert replace_static(scene,bank,True) is scene and scene.static is bank and scene.dynamic is dynamic
    assert all(torch.equal(v,before[k]) for k,v in state.items())


@pytest.mark.parametrize('error',['shape','nonfinite','duplicate','missing','extra','empty'])
def test_reject_bad_bank(error):
    s=payload(0 if error=='empty' else 3)
    if error=='shape':s['_scaling']=torch.zeros(3,2)
    if error=='nonfinite':s['_xyz'][0,0]=float('nan')
    if error=='duplicate':s['point_ids'][1]=s['point_ids'][0]
    if error=='missing':del s['initial_xyz']
    if error=='extra':s['semantic_labels']=torch.zeros(3)
    with pytest.raises(ValueError):validate_payload(s)


def test_missing_explicit_source():
    with pytest.raises(ValueError):replace_static(SimpleNamespace(static=None),enabled=True)
