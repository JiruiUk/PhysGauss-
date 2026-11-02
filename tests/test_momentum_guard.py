import pytest
import torch
from variants.momentum_guard.balance import balance


def test_cap_bounds_each_shared_group_and_does_not_differentiate_scale():
    a=torch.tensor([1.,2.],requires_grad=True); b=torch.tensor([3.],requires_grad=True)
    head=torch.tensor([.3],requires_grad=True)
    image=a.sum()+10*b.sum()
    physical=1000*a.sum()+b.sum()+head.sum()
    loss, info=balance(image,physical,{'grid':[a],'deform':[b]},[head],.1,.01)
    assert info['scale']==pytest.approx(.01)
    gs=torch.autograd.grad(.01*loss,[a,b,head])
    assert torch.allclose(gs[0],torch.full_like(a,.1))
    assert gs[1].item()==pytest.approx(.0001)
    assert gs[2].item()==pytest.approx(.01)
    assert loss.item()==physical.item()
    for row in info['groups'].values():
        assert row['guarded_momentum_norm'] <= .1*row['image_norm']*(1+1e-6)


def test_small_momentum_is_exact_no_op():
    x=torch.tensor([1.],requires_grad=True)
    image=10*x.sum(); physical=.1*x.sum()
    guarded,info=balance(image,physical,{'shared':[x]},[],.1,.01)
    assert info['scale']==1 and guarded is physical


def test_zero_image_reference_suppresses_only_this_physics_event():
    x=torch.tensor([1.],requires_grad=True)
    loss,info=balance(x.sum()*0,x.square().sum(),{'shared':[x]},[],.1,.01)
    assert info['scale']==0
    assert torch.autograd.grad(loss,x)[0].item()==0


def test_unused_parameters_are_supported_and_nonfinite_is_rejected():
    x=torch.tensor([1.],requires_grad=True); unused=torch.tensor([2.],requires_grad=True)
    _,info=balance(x.sum(),x.square().sum(),{'shared':[x,unused]},[],.1,.01)
    assert info['scale']==1
    with pytest.raises(FloatingPointError):
        balance(x.sum(),x.sum()*float('nan'),{'shared':[x]},[],.1,.01)
    with pytest.raises(ValueError):
        balance(x.sum(),x.sum(),{'shared':[x]},[],0,.01)


def test_guard_off_returns_exact_parent_tensors(tmp_path, monkeypatch):
    import json
    from variants.momentum_guard.runtime import Tuning, Parent
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'enabled':True, 'momentum_gradient_guard':False}))
    tuning=Tuning(config,tmp_path)
    value=torch.tensor(2.,requires_grad=True)
    monkeypatch.setattr(Parent,'image_loss',lambda *args:value)
    monkeypatch.setattr(Parent,'momentum_loss',lambda *args:value)
    assert tuning.image_loss(100,None,None,None) is value
    assert tuning.momentum_loss(100,None,None,None) is value
    assert tuning._reference is None


def test_guard_rejects_missing_same_step_reference(tmp_path, monkeypatch):
    import json
    from variants.momentum_guard.runtime import Tuning, Parent
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'enabled':True, 'momentum_gradient_guard':True,
        'stress_source':'shared','velocity_source':'trajectory','material_momentum':True,'momentum_start':30}))
    tuning=Tuning(config,tmp_path)
    monkeypatch.setattr(Parent,'momentum_loss',lambda *args:torch.tensor(2.,requires_grad=True))
    with pytest.raises(RuntimeError,match='same-iteration'):
        tuning.momentum_loss(100,None,None,None)
