import math
import pytest
import torch
from types import SimpleNamespace
from variants.staged_reconstruction.invertible_motion.core import InvertibleMotion,install,InvertibleDeform


def perturbed():
    with torch.random.fork_rng():
        torch.manual_seed(712)
        model=InvertibleMotion(width=12).double()
        with torch.no_grad():
            for layer in model.layers:
                layer.conditioner[-1].weight.normal_(0,.3)
                layer.conditioner[-1].bias.normal_(0,.3)
    return model


def test_zero_heads_are_exact_identity():
    model=InvertibleMotion(width=12).double()
    x=torch.tensor([[.2,.3,-.1],[-.4,.1,.7]],dtype=torch.double)
    y,logdet=model(x,x.new_tensor([[0.],[1.]]),return_logdet=True)
    assert torch.equal(x,y) and torch.equal(logdet,torch.zeros_like(logdet))


@pytest.mark.parametrize('time',[0.,.43,1.])
def test_round_trip_and_positive_analytic_determinant(time):
    model=perturbed(); x=torch.tensor([[.2,.3,-.1],[-.4,.1,.7]],dtype=torch.double)
    t=x.new_full((2,1),time)
    y,logdet=model(x,t,return_logdet=True)
    restored,inverse_logdet=model(y,t,inverse=True,return_logdet=True)
    assert torch.allclose(restored,x,atol=2e-12)
    assert torch.allclose(logdet,-inverse_logdet,atol=2e-12)
    jac=torch.autograd.functional.jacobian(lambda point:model(point[None],t[:1])[0],x[0])
    assert torch.allclose(torch.linalg.det(jac),logdet[0].exp(),atol=1e-12)
    assert math.exp(-1.2)<=float(logdet[0].exp().detach())<=math.exp(1.2)


def test_spatial_time_and_parameter_gradients_are_finite():
    model=perturbed(); x=torch.tensor([[.2,.3,-.1]],dtype=torch.double,requires_grad=True)
    t=torch.tensor([[.43]],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a,b:model(a,b),(x,t),eps=1e-6,atol=1e-5)
    loss=model(x,t).square().sum(); gradients=torch.autograd.grad(loss,[x,t,*model.parameters()])
    assert all(g.isfinite().all() for g in gradients)
    assert all(g.abs().sum()>0 for g in gradients)


def test_reject_wrong_time_shape():
    model=InvertibleMotion(width=12)
    with pytest.raises(ValueError): model(torch.zeros(2,3),torch.zeros(1,1))


def test_disabled_install_is_noop_even_without_configuration():
    original=object(); scene=SimpleNamespace(deforms=[original])
    assert install(scene,{},False) is None and scene.deforms[0] is original
    assert install(scene,{'invertible_motion_enabled':False},True) is None
    assert scene.deforms[0] is original


def test_install_preserves_rng_and_original_optimizer_group():
    original=torch.nn.Linear(3,3)
    scene=SimpleNamespace(deforms=[object()],networks=torch.nn.ModuleList([original]),centers=torch.zeros(1,3))
    opt=torch.optim.Adam([{'params':original.parameters(),'lr':.001,'name':'original'}])
    rng=torch.get_rng_state().clone(); ids=[id(p) for p in opt.param_groups[0]['params']]
    plan=dict(invertible_motion_enabled=True,initialization_seed=712,width=12,
              log_scale_bound=.2,translation_bound=.2,learning_rate=.0008)
    adapter=install(scene,plan,True,opt)
    assert isinstance(adapter,InvertibleDeform) and torch.equal(rng,torch.get_rng_state())
    assert ids==[id(p) for p in opt.param_groups[0]['params']]
    assert len(opt.param_groups)==2 and opt.param_groups[-1]['name']=='invertible_motion'
    assert len(scene.networks)==2
    with pytest.raises(ValueError): install(scene,plan,True,opt)
