import json
from types import SimpleNamespace
import pytest
import torch
from variants.coverage_isolated.runtime import Tuning
from variants.coverage_tuning.runtime import Tuning as Original


def runtime(tmp_path,**config):
    p = tmp_path/'config.json'; p.write_text(json.dumps(dict(enabled=True,**config)))
    result = Tuning(p,tmp_path); result.train_names = {'0002'}
    return result


def test_affine_only_loss_and_gradient_match_original_without_accessing_alpha(tmp_path):
    config = dict(affine_weight=1e-4,mask_weight=0.,coverage_density=False,
        affine_start=0,affine_ramp=1,affine_samples=12,ack_unsegmented_affine=True)
    a = runtime(tmp_path,**config)
    p = tmp_path/'original.json'; p.write_text(json.dumps(dict(a.config,single_object=True)))
    b = Original(p,tmp_path); b.train_names = a.train_names
    g = torch.Generator().manual_seed(8)
    x = torch.randn(50,3,generator=g)*.01
    w = torch.cat([torch.tensor([[1.,0.,0.,0.]]*50),torch.randn(50,3,generator=g)*.01],-1)
    first = w.clone().requires_grad_(); second = w.clone().requires_grad_()
    xyz = SimpleNamespace(get_xyz=x)
    # New path needs neither render package nor a GT coverage_target attribute.
    la = a.coverage_loss(100,SimpleNamespace(image_name='0002'),None,xyz,first)
    lb = b.coverage_loss(100,SimpleNamespace(image_name='0002',coverage_target=torch.zeros(1,2,2)),
        {'alpha':torch.zeros(1,2,2)},xyz,second)
    la.backward(); lb.backward()
    assert torch.allclose(la,lb,atol=1e-12) and la>0
    assert torch.allclose(first.grad,second.grad,atol=1e-10)
    assert first.grad.abs().sum()>0 and torch.isfinite(first.grad).all()


def test_density_alone_has_no_differentiable_mask_or_affine_loss(tmp_path):
    a = runtime(tmp_path,coverage_density=True,mask_weight=0.,affine_weight=0.)
    calls = []
    a.observe_coverage = lambda *args:calls.append(args)
    alpha = torch.rand(1,2,2,requires_grad=True)
    result = a.coverage_loss(20,SimpleNamespace(image_name='0002',coverage_target=torch.ones(1,2,2)),
        {'alpha':alpha},SimpleNamespace(get_xyz=torch.zeros(5,3)),None)
    assert result is None and len(calls)==1 and alpha.grad is None


def test_isolated_scope_and_flags_off(tmp_path):
    with pytest.raises(ValueError): runtime(tmp_path,mask_weight=.1)
    with pytest.raises(ValueError): runtime(tmp_path,affine_weight=1e-4)
    with pytest.raises(ValueError): runtime(tmp_path,affine_weight=1e-4,coverage_density=True,ack_unsegmented_affine=True)
    p = tmp_path/'off.json'; p.write_text(json.dumps({'enabled':False}))
    off = Tuning(p,tmp_path); before = torch.get_rng_state().clone()
    assert off.coverage_loss(1,None,None,None,None) is None
    off.after_update(1,None)
    assert torch.equal(before,torch.get_rng_state())


def test_component_configs_keep_exact_scratch_optimizer_and_no_extra_modules():
    from variants.coverage_isolated.run import configuration,NAMES
    from variants.coverage_tuning.run import configuration as source
    reference,text = source('C0_V3_reference')
    a,ta = configuration(NAMES[0]); b,tb = configuration(NAMES[1])
    assert ta==tb==text
    for c in (a,b):
        assert c['mask_weight']==c['kinematic_weight']==c['isotropic_weight']==0
        assert c['parent_run']=='' and 'single_object' not in c
    assert a['affine_weight']==1e-4 and a['coverage_density'] is False
    assert b['affine_weight']==0 and b['coverage_density'] is True


def test_fork_only_changes_tuning_import():
    from variants.coverage_isolated.prepare import prepare
    from variants.coverage_tuning.prepare import prepare as parent
    text = (prepare()/'train_pidg.py').read_text(encoding='utf-8')
    assert text.replace('from variants.coverage_isolated.runtime import Tuning',
        'from variants.coverage_tuning.runtime import Tuning')==(parent()/'train_pidg.py').read_text(encoding='utf-8')
