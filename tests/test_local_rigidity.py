import numpy as np
import pytest
import torch
from variants.staged_reconstruction.local_rigidity.core import canonical_graph,quaternion_matrix,coherence


def fixture():
    x=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=torch.double)
    edges=torch.tensor([[0,1],[1,0],[0,2],[2,0],[0,3],[3,0]])
    return x,edges,torch.ones(6,dtype=torch.double),torch.ones(6,dtype=torch.double)


def test_global_rigid_transform_has_zero_loss():
    x,e,l,w=fixture()
    r=quaternion_matrix(torch.tensor([[.8,.2,-.3,.4]],dtype=x.dtype)).repeat(4,1,1)
    p=(r@x[...,None]).squeeze(-1)+torch.tensor([3.,-2.,5.],dtype=x.dtype)
    terms=coherence(x,p,r,e,l,w)
    assert terms['rigid']<1e-28 and terms['rotation']==0


def test_scale_change_is_penalized():
    x,e,l,w=fixture(); r=torch.eye(3,dtype=x.dtype).repeat(4,1,1)
    assert coherence(x,2*x,r,e,l,w)['rigid']>.01


def test_neighbor_rotation_disagreement_penalized():
    x,e,l,w=fixture(); q=torch.tensor([[1.,0.,0.,0.]]*4,dtype=x.dtype); q[1]=torch.tensor([0.,0.,0.,1.])
    assert coherence(x,x,quaternion_matrix(q),e,l,w)['rotation']>.01


def test_backpropagates_into_position_canonical_rotation():
    x,e,l,w=fixture(); x.requires_grad_(); p=(x.detach()*1.1).requires_grad_()
    q=torch.tensor([[1.,.05,0.,0.]]*4,dtype=x.dtype,requires_grad=True)
    terms=coherence(x,p,quaternion_matrix(q),e,l,w); sum(terms.values()).backward()
    assert all(v.grad is not None and v.grad.isfinite().all() and v.grad.abs().sum()>0 for v in [x,p,q])


def test_gradcheck():
    x,e,l,w=fixture(); p=(1.1*x).requires_grad_(); x.requires_grad_()
    q=torch.tensor([[1.,.02,.01,.03]]*4,dtype=x.dtype,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a,b,c:sum(coherence(a,b,quaternion_matrix(c),e,l,w).values()),(x,p,q),eps=1e-6,atol=1e-5)


def test_graph_never_bridges_distant_patches():
    rng=np.random.default_rng(0); x=np.r_[rng.normal(0,.01,(20,3)),rng.normal(2,.01,(20,3))]
    e,l,w=canonical_graph(x,k=10,radius=.08)
    assert not np.any((e[:,0]<20)!=(e[:,1]<20))
    assert np.all(l>0) and np.all(w>0) and len(np.unique(e))==40


def test_quaternion_sign_and_scale_do_not_change_rotation():
    q=torch.tensor([[1.,.2,.3,.4]],dtype=torch.double)
    assert torch.allclose(quaternion_matrix(q),quaternion_matrix(-3*q))


def test_empty_graph_rejected():
    with pytest.raises(ValueError): canonical_graph(np.array([[0.,0.,0.],[5.,5.,5.]]),radius=.01)
