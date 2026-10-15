import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pidg_ext.deformation_gradient import canonical_neighbors, estimate_F, CanonicalGraph


def cloud():
    return torch.randn(32,3,generator=torch.Generator().manual_seed(81),dtype=torch.float64)


@pytest.mark.parametrize('kind',['identity','scale','rotation','affine'])
def test_local_F_recovers_known_affine_transform(kind):
    X=cloud();A=torch.eye(3,dtype=X.dtype)
    if kind=='scale':A*=1.3
    if kind=='rotation':A=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=X.dtype)
    if kind=='affine':A=torch.tensor([[1.2,.1,0.],[.2,.9,.1],[0.,.1,1.1]],dtype=X.dtype)
    result=estimate_F(X,X@A.T+.7,canonical_neighbors(X),regularization=1e-7)
    assert result['valid'].all()
    torch.testing.assert_close(result['F'],A.expand(32,3,3),atol=2e-6,rtol=2e-6)


def test_F_autograd_degenerate_mask_and_negative_determinant():
    X=cloud();graph=canonical_neighbors(X);x=(X*1.1).requires_grad_()
    result=estimate_F(X,x,graph);result['F'].square().sum().backward()
    assert x.grad.isfinite().all() and x.grad.norm()>0
    reflected=X.clone();reflected[:,0]*=-1
    result=estimate_F(X,reflected,graph)
    assert (result['J']<0).all() and not result['valid'].any()
    flat=X.clone();flat[:,2]=0
    result=estimate_F(flat,flat,canonical_neighbors(flat))
    assert not result['valid'].any() and result['F'].isfinite().all()


def test_F_backward_matches_finite_differences():
    X=cloud()[:10];graph=canonical_neighbors(X,8)
    x=(X*1.1).requires_grad_()
    assert torch.autograd.gradcheck(lambda y:estimate_F(X,y,graph)['F'],(x,),fast_mode=True)


def test_graph_cache_does_not_consume_training_rng_and_refreshes_on_topology_change():
    import numpy as np
    import random
    X=cloud();rng=torch.random.get_rng_state().clone();cache=CanonicalGraph()
    numpy_rng=np.random.get_state();python_rng=random.getstate()
    first=cache.get(X,1)
    assert cache.get(X,2) is first
    assert len(cache.get(X[:20],3))==20
    assert torch.equal(rng,torch.random.get_rng_state())
    assert random.getstate()==python_rng
    for before,after in zip(numpy_rng,np.random.get_state()):
        np.testing.assert_equal(before,after)


def test_disabled_extension_is_a_noop_and_does_not_consume_rng(tmp_path):
    from pidg_ext.runtime import Extension
    rng=torch.random.get_rng_state().clone()
    extension=Extension(None,tmp_path)
    assert not extension.enabled
    assert extension.loss(100,None,None,None,None,None) is None
    assert torch.equal(rng,torch.random.get_rng_state())
    assert not list(tmp_path.iterdir())


def test_neo_hookean_identity_rotation_expansion_and_objectivity():
    from pidg_ext.constitutive import NeoHookean
    model=NeoHookean();params={'mu':1.,'kappa':10.};I=torch.eye(3,dtype=torch.float64)
    R=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=I.dtype)
    torch.testing.assert_close(model(I,params),torch.zeros_like(I))
    torch.testing.assert_close(model(R,params),torch.zeros_like(I))
    stress=model(I*1.1,params)
    torch.testing.assert_close(stress,stress[0,0]*I)
    F=I.clone();F[0,0]=1.2;F[0,1]=.1
    torch.testing.assert_close(model(R@F,params),R@model(F,params)@R.T)


def test_neo_hookean_parameter_gradients_and_symmetric_component_order():
    from pidg_ext.constitutive import NeoHookean,MaterialParameters,vec6_to_symmat,symmat_to_vec6
    bank=MaterialParameters(1.,10.,learn=True)
    F=torch.diag(torch.tensor([1.2,1.1,1.05],dtype=torch.float64)).requires_grad_()
    NeoHookean()(F,bank()).square().sum().backward()
    assert F.grad.isfinite().all()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in bank.parameters())
    assert all(p.grad.abs().sum()>0 for p in bank.parameters())
    assert all(value>0 for value in bank().values())
    vec=torch.tensor([1.,2.,3.,4.,5.,6.])
    mat=vec6_to_symmat(vec)
    torch.testing.assert_close(mat,torch.tensor([[1.,4.,5.],[4.,2.,6.],[5.,6.,3.]]))
    torch.testing.assert_close(symmat_to_vec6(mat),vec)
    with pytest.raises(ValueError,match='positive determinant'):
        NeoHookean()(-torch.eye(3),bank())


def test_constitutive_loss_reaches_deformed_positions_predicted_stress_and_shared_parameters():
    from pidg_ext.constitutive import MaterialParameters
    from pidg_ext.losses import constitutive_consistency,predicted_stress
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__();self.v=torch.nn.Parameter(torch.tensor(.2,dtype=torch.float64))
            self.s=torch.nn.Parameter(torch.zeros(6,dtype=torch.float64))
        def forward(self,coords,indices):
            return coords[:,:3].square()*self.v,self.s.expand(len(coords),6)
    X=cloud();x=(X@torch.diag(torch.tensor([1.2,.95,1.1],dtype=X.dtype))).requires_grad_()
    result=estimate_F(X,x,canonical_neighbors(X))
    field=Field();coords=torch.cat((X,torch.ones(32,1,dtype=X.dtype)),dim=-1).requires_grad_()
    force_sigma=torch.tensor(.3,dtype=X.dtype,requires_grad=True)
    sigma=predicted_stress(coords,torch.arange(32),field,force_sigma)
    bank=MaterialParameters(mu=.5,kappa=.8,learn=True)
    raw,_=constitutive_consistency(sigma,result['F'],bank(),result['confidence'],result['valid'])
    raw.backward()
    for tensor in [x,field.v,field.s,force_sigma,*bank.parameters()]:
        assert tensor.grad is not None and tensor.grad.isfinite().all() and tensor.grad.norm()>0


def test_invalid_F_is_explicitly_masked_from_constitutive_loss():
    from pidg_ext.losses import constitutive_consistency
    F=(-torch.eye(3)).expand(2,3,3).clone().requires_grad_()
    pred=torch.ones_like(F,requires_grad=True)
    raw,stress=constitutive_consistency(pred,F,{'mu':1.,'kappa':10.},torch.ones(2),torch.zeros(2,dtype=torch.bool))
    assert raw.item()==0 and not stress.any()
    raw.backward();assert not pred.grad.any()


def test_extension_restart_restores_material_optimizer_and_cached_graph(tmp_path):
    import json
    from pidg_ext.runtime import Extension
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'deformation_gradient':True,'constitutive':True,'learn_material_parameters':True}))
    original=Extension(config,tmp_path,150)
    def step(extension):
        params=extension.material()
        ((params['mu']-1.2).square()+(params['kappa']-8.).square()).backward()
        extension.optimizer_step()
    original.graph.get(cloud(),100)
    step(original);saved=original.checkpoint_state()
    resumed=Extension(config,tmp_path,150);resumed.restore_checkpoint(saved)
    step(original);step(resumed)
    for left,right in zip(original.material.parameters(),resumed.material.parameters()):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    assert torch.equal(original.graph.neighbors,resumed.graph.neighbors)
    assert resumed.graph.last_iteration==100


def test_constitutive_stress_uses_same_physical_time_as_rendered_deformation(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from pidg_ext.runtime import Extension
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__();self.scale=torch.nn.Parameter(torch.tensor(.1,dtype=torch.float64))
        def forward(self,coords,indices):
            self.observed_times=coords[:,3].detach()
            return coords[:,:3].square()*self.scale, self.scale.expand(len(coords),6)
    monkeypatch.setitem(sys.modules,'utils.general_utils',SimpleNamespace(quaternion_to_matrix=None))
    config=tmp_path/'config.json';config.write_text(json.dumps({'deformation_gradient':True,'constitutive':True}))
    field=Field();g=SimpleNamespace(get_xyz=cloud().requires_grad_(),particle_indices=torch.arange(32),velocity_net=field,force_sigma=torch.tensor(.3))
    ext=Extension(config,tmp_path,150)
    loss=ext.loss(100,g,None,0.,torch.tensor([.73],dtype=torch.float64),None)
    torch.testing.assert_close(field.observed_times,torch.full((32,),.73,dtype=torch.float64))
    assert loss.isfinite()


def test_fixed_and_learnable_ablation_changes_only_parameter_fitting():
    import json
    root=Path(__file__).resolve().parents[1]/'experiments/configs'
    fixed=json.loads((root/'A2a.json').read_text());learned=json.loads((root/'A2b.json').read_text())
    assert not fixed['learn_material_parameters'] and learned['learn_material_parameters']
    for config in (fixed,learned):
        for key in ('experiment','learn_material_parameters','material_lr'):config.pop(key,None)
    assert fixed==learned
