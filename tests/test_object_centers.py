"""Object-center primitives and fixed candidate grouping before integration."""
import numpy as np
import pytest
import torch
from variants.object_centers.model import weighted_centers, center_residual, CenterSpline, inherit_groups, cross_group_sphere_proxy
from variants.object_centers.grouping import cluster_candidates


def test_weighted_centers_and_local_residual_remove_translation_gauge():
    x=torch.tensor([[0.,0.,0.],[2.,0.,0.],[10.,1.,0.],[12.,1.,0.]],requires_grad=True)
    ids=torch.tensor([0,0,1,1]);w=torch.tensor([1.,3.,1.,1.])
    centers,residual=center_residual(x,ids,w,2)
    assert torch.allclose(centers,torch.tensor([[1.5,0,0],[11.,1.,0.]]))
    assert torch.allclose(weighted_centers(residual,ids,w,2),torch.zeros(2,3),atol=1e-6)
    assert torch.equal(centers[ids]+residual,x)
    centers.sum().backward();assert x.grad.abs().sum()>0


def test_center_trajectory_interpolates_and_learns_group_motion():
    times=torch.tensor([0.,.5,1.],dtype=torch.float64)
    initial=torch.tensor([[[0.,0.,0.]],[[1.,0.,0.]],[[2.,0.,0.]]],dtype=torch.float64)
    model=CenterSpline(times,initial)
    assert torch.allclose(model(torch.tensor([.25],dtype=torch.float64)),torch.tensor([[[.5,0.,0.]]],dtype=torch.float64))
    assert model.smoothness()<1e-12
    model(torch.tensor([.25],dtype=torch.float64)).sum().backward()
    assert model.offsets.grad.abs().sum()>0


def test_split_children_inherit_membership_and_pruning_preserves_alignment():
    labels=torch.tensor([0,1,-1])
    parents=torch.tensor([1,0]);keep=torch.tensor([False,True,True,True,True])
    assert torch.equal(inherit_groups(labels,parents,keep),torch.tensor([1,-1,1,0]))


def test_contact_proxy_ignores_same_group_unknown_and_separated_pairs():
    x=torch.tensor([[0.,0.,0.],[.1,0.,0.],[10.,0.,0.]],requires_grad=True)
    pairs=torch.tensor([[0,1],[0,2]])
    assert cross_group_sphere_proxy(x,torch.ones(3),torch.tensor([0,0,1]),pairs)==0
    assert cross_group_sphere_proxy(x,torch.ones(3),torch.tensor([0,-1,1]),pairs)==0


def test_overlap_proxy_has_finite_separating_gradients_even_at_coincidence():
    x=torch.zeros(2,3,requires_grad=True);r=torch.ones(2,requires_grad=True)
    loss=cross_group_sphere_proxy(x,r,torch.tensor([0,1]),torch.tensor([[0,1]]))
    loss.backward()
    assert loss>0 and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert torch.allclose(x.grad[0],-x.grad[1]) and r.grad is None


def test_missing_group_is_explicitly_rejected():
    with pytest.raises(ValueError,match='empty'):
        weighted_centers(torch.zeros(3,3),torch.zeros(3,dtype=torch.long),torch.ones(3),2)


def test_invalid_group_id_is_rejected_before_residual_indexing():
    with pytest.raises(ValueError,match='Group IDs'):
        center_residual(torch.zeros(2,3),torch.tensor([0,2]),torch.ones(2),2)


@pytest.mark.parametrize('mode',['canonical','trajectory','motion_dominant'])
def test_fixed_trajectory_clustering_is_deterministic_and_leaves_unknown_ids(mode):
    rng=np.random.default_rng(4);n=80
    x=np.concatenate([rng.normal(0,.02,(n,3))+[-1,0,0],rng.normal(0,.02,(n,3))+[1,0,0]])
    poses=np.stack([x,x+np.r_[np.tile([0,.4,0],(n,1)),np.tile([0,-.4,0],(n,1))]])
    colors=np.tile([.8,.7,.1],(2*n,1));eligible=np.ones(2*n,dtype=bool);eligible[-1]=False
    a=cluster_candidates(x,poses,colors,eligible,np.ones(2*n),2,mode)
    b=cluster_candidates(x,poses,colors,eligible,np.ones(2*n),2,mode)
    assert np.array_equal(a,b) and a[-1]==-1
    assert len(set(a[:n]))==len(set(a[n:-1]))==1 and a[0]!=a[n]
