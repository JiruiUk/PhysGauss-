"""Specify stable color and train-foreground objectives before integration."""
import pytest
import torch
import json
from types import SimpleNamespace
from variants.color_repair.model import positive_color, inverse_positive_color, foreground_rgb_loss


def test_inverse_initialization_preserves_original_positive_colors():
    x=torch.tensor([.0001,.003,.02,.1,.5,1.],dtype=torch.float64)
    assert torch.allclose(positive_color(inverse_positive_color(x)),x,atol=1e-12)


def test_negative_channel_has_positive_finite_gradient():
    q=torch.tensor([-.5,-.1,0.,.5],dtype=torch.float64,requires_grad=True)
    c=positive_color(q)
    c.sum().backward()
    assert (c>0).all() and (q.grad>0).all() and torch.isfinite(q.grad).all()
    assert torch.autograd.gradcheck(positive_color,(q,))


def test_softplus_large_positive_input_is_numerically_stable():
    assert torch.isfinite(positive_color(torch.tensor([100.,1000.]))).all()


def test_foreground_rgb_loss_ignores_background_and_detaches_targets():
    image=torch.zeros(3,2,2,requires_grad=True)
    gt=torch.ones(3,2,2,requires_grad=True)
    mask=torch.tensor([[[1.,0.],[0.,0.]]],requires_grad=True)
    loss=foreground_rgb_loss(image,gt,mask)
    assert loss.item()==pytest.approx(1.)
    loss.backward()
    assert image.grad[:,0,0].abs().sum()>0 and image.grad[:,1,1].abs().sum()==0
    assert gt.grad is None and mask.grad is None


def test_foreground_rgb_loss_is_invariant_to_background_padding():
    im=torch.zeros(3,2,2);gt=torch.ones(3,2,2);mask=torch.ones(1,2,2)
    import torch.nn.functional as F
    assert foreground_rgb_loss(im,gt,mask)==foreground_rgb_loss(F.pad(im,(0,5,0,5)),F.pad(gt,(0,5,0,5)),F.pad(mask,(0,5,0,5)))


def test_empty_foreground_is_zero_finite():
    im=torch.ones(3,2,2,requires_grad=True)
    loss=foreground_rgb_loss(im,torch.zeros_like(im),torch.zeros(1,2,2))
    loss.backward();assert loss==0 and im.grad.abs().sum()==0


@pytest.mark.parametrize('beta',[0.,-1.,float('nan')])
def test_invalid_beta_rejected(beta):
    with pytest.raises(ValueError):positive_color(torch.ones(3),beta)


def test_color_evidence_recovers_mean_variance_and_roundtrips():
    from variants.color_repair.evidence import ColorEvidence
    evidence=ColorEvidence(3)
    contributors=torch.tensor([[[0,1]],[[0,-1]]])
    deficit=torch.tensor([[[1.,2.]]]);rgb=torch.tensor([[[.2,.8]],[[.3,.7]],[[.4,.6]]])
    evidence.observe(contributors,deficit,rgb)
    means,std=evidence.mean_std(torch.tensor([0,1]))
    assert torch.allclose(means,torch.tensor([[.2,.3,.4],[.8,.7,.6]]))
    assert std.max()<1e-3
    clone=ColorEvidence.from_state(evidence.state_dict(),'cpu')
    assert torch.equal(clone.first,evidence.first) and torch.equal(clone.mass,evidence.mass)


def make_runtime(tmp_path,**extra):
    from variants.color_repair.runtime import Tuning
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(enabled=True,mask_weight=.03,foreground_rgb_weight=.1,mask_start=20,mask_ramp=10,**extra)))
    runtime=Tuning(config,tmp_path);runtime.train_names={'0002'}
    return runtime


def test_repaired_mask_is_delayed_ramped_and_training_only(tmp_path):
    runtime=make_runtime(tmp_path)
    view=SimpleNamespace(image_name='0002',coverage_target=torch.ones(1,3,4),original_image=torch.ones(3,3,4))
    alpha=torch.zeros(1,3,4,requires_grad=True);rgb=torch.zeros(3,3,4,requires_grad=True)
    package={'alpha':alpha,'render':rgb}
    assert runtime.coverage_loss(20,view,package,None,None) is None
    loss=runtime.coverage_loss(21,view,package,None,None)
    assert loss.item()==pytest.approx(.1*(.03+.1))
    loss.backward();assert rgb.grad.abs().sum()>0 and alpha.grad.abs().sum()>0
    with pytest.raises(ValueError,match='Only training'):
        runtime.coverage_loss(21,SimpleNamespace(image_name='0001'),package,None,None)


def test_flags_off_returns_original_path_without_alpha_or_rng_access(tmp_path):
    from variants.color_repair.runtime import Tuning
    config=tmp_path/'off.json';config.write_text(json.dumps({'enabled':False}))
    runtime=Tuning(config,tmp_path);state=torch.get_rng_state().clone()
    assert runtime.coverage_loss(50,None,None,None,None) is None
    runtime.after_update(50,None)
    assert torch.equal(state,torch.get_rng_state())


def test_initial_softplus_conversion_preserves_random_colors(tmp_path):
    from variants.color_repair.runtime import Tuning
    config=tmp_path/'config.json';config.write_text(json.dumps({'enabled':True,'positive_color':True}))
    runtime=Tuning(config,tmp_path)
    dc=torch.nn.Parameter(torch.tensor([[[.003,.02,.5]]]))
    old=dc.detach().clone()*.28209479177387814+.5
    gaussian=SimpleNamespace(_features_dc=dc,_features_rest=torch.zeros(1,15,3))
    runtime.after_scene(gaussian,None)
    assert torch.allclose(positive_color(dc*.28209479177387814+.5),old,atol=1e-6)


def test_repair_preserves_V3_protocol_and_has_conservative_density():
    from variants.color_repair.run import configuration,NAMES
    a,ta=configuration(NAMES[0]);b,tb=configuration(NAMES[1]);c,tc=configuration(NAMES[2])
    assert ta==tb==tc
    for config in (a,b,c):
        assert config['positive_color'] and config['mask_weight']==.03
        assert config['mask_start']==3000 and config['mask_ramp']==2000
        assert config['foreground_rgb_weight']==.1 and config['perceptual_weight']==.02
        assert config['shared_physics_time'] and config['timeline']=='frame_index'
        assert config['kinematic_weight']==config['isotropic_weight']==config['centroid_weight']==0
        assert config['parent_run']=='' and 'single_object' not in config
    assert a['affine_weight']==0 and b['affine_weight']==c['affine_weight']==5e-5
    assert b['affine_radius']==c['affine_radius']==.05
    assert not a['coverage_density'] and not b['coverage_density'] and c['coverage_density']
    assert c['coverage_add_limit']==256


def test_fork_preserves_parent_and_only_adds_optional_color_branch():
    from variants.color_repair.prepare import prepare
    from variants.coverage_tuning.prepare import prepare as parent
    new=prepare();old=parent()
    assert (new/'train_pidg.py').read_text().replace('from variants.color_repair.runtime import Tuning',
        'from variants.coverage_tuning.runtime import Tuning')==(old/'train_pidg.py').read_text()
    assert 'positive_colors(pc, means3D, viewpoint_camera)' in (new/'gaussian_renderer/__init__.py').read_text()
