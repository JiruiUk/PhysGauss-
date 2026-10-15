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
