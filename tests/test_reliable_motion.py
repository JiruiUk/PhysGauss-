"""Rigid transforms, covariances, directed gradients and honest missing data."""
import numpy as np
import pytest
import torch
from variants.staged_reconstruction.reliable_motion.core import build_patches, fit_local_rotation, propagate, extend


def example():
    xyz = np.array([[-.02,-.02,0.], [.02,-.02,0.], [.02,.02,0.], [-.02,.02,0.], [0.,0.,.015], [0.,0.,.007]], np.float32)
    x = np.repeat(xyz[None], 6, axis=0)
    s = np.ones(x.shape[:2]); q = np.ones_like(s); op = np.ones(6)
    keys = np.arange(5); trusted = np.zeros_like(s, dtype=bool); trusted[:, keys] = True
    return x, s, q, op, keys, trusted


def torch_graph():
    raw = build_patches(*example(), radius=.08, candidates=5, neighbors=4)
    g = {k: torch.as_tensor(v, dtype=torch.long if np.asarray(v).dtype.kind in 'iu' else torch.float64)
         for k,v in raw.items()}
    g['key_lookup'] = torch.zeros(6,dtype=torch.long); g['key_lookup'][g['keys']] = torch.arange(5)
    g['reference_covariance'] = torch.diag(torch.tensor([.0002,.0001,.00005], dtype=torch.float64)).repeat(len(g['targets']),1,1)
    g['camera_rotation'] = torch.eye(3,dtype=torch.float64).repeat(6,1,1)
    return g


def test_proper_rigid_rotation_is_exact_and_covariance_transforms_consistently():
    g=torch_graph(); x=torch.tensor(example()[0][0],dtype=torch.float64)
    angle=torch.tensor(.9,dtype=torch.float64)
    r=torch.tensor([[angle.cos(),-angle.sin(),0.],[angle.sin(),angle.cos(),0.],[0.,0.,1.]],dtype=torch.float64)
    transformed=x@r.T+torch.tensor([.1,-.2,.3])
    c=g['reference_covariance'][0].repeat(len(x),1,1)
    c=r@c@r.T
    terms,_=propagate(transformed,c,g,0,torch.arange(len(g['targets'])))
    assert terms['center'] < 1e-8 and terms['covariance'] < 1e-8
    assert terms['tether'] > 0  # Observation tether is intentionally not global-transform invariant.


def test_planar_repeated_singular_values_have_finite_directed_gradients():
    g=torch_graph(); x=torch.tensor(example()[0][0],dtype=torch.float64,requires_grad=True)
    x2=x+torch.tensor([[0.,0.,0.]]*5+[[.01,0.,0.]],dtype=torch.float64)
    c=g['reference_covariance'][0].repeat(len(x),1,1).clone().requires_grad_(True)
    rows=torch.nonzero(g['targets']==5).flatten()
    terms,_=propagate(x2,c,g,0,rows)
    (terms['center']+terms['covariance']+terms['tether']).backward()
    assert x.grad.isfinite().all() and x.grad[5].norm()>0 and x.grad[:5].norm()>0
    assert c.grad.isfinite().all()


def test_detached_local_rotation_is_proper_even_for_reflection():
    a=torch.tensor(example()[0][0,:4],dtype=torch.float64)[None]
    b=a.clone();b[...,0]*=-1
    r,_,_=fit_local_rotation(a,b,torch.ones(1,4,dtype=torch.float64))
    assert not r.requires_grad
    torch.testing.assert_close(torch.linalg.det(r),torch.ones(1,dtype=torch.float64))


def test_collinear_or_missing_reference_patches_are_rejected():
    x,s,q,op,keys,tr=example();x[...,1:]=0
    with pytest.raises(ValueError,match='non-collinear'):
        build_patches(x,s,q,op,keys,tr,radius=.08,candidates=5,neighbors=4)
    tr[:]=False
    with pytest.raises(ValueError,match='supported'):
        build_patches(x,s,q,op,keys,tr,radius=.08,candidates=5,neighbors=4)


def test_contact_only_directed_pairs_do_not_survive_common_time_radius():
    x,s,q,op,keys,tr=example();x[3:,4,0]+=1
    g=build_patches(x,s,q,op,keys,tr,radius=.08,candidates=5,neighbors=4)
    active_sources=g['sources'][g['spatial_weights']>0]
    assert not np.any(active_sources==4)


def test_insufficient_common_observations_do_not_get_filled_to_neighbor_budget():
    x,s,q,op,keys,tr=example();tr[2:,4]=False
    g=build_patches(x,s,q,op,keys,tr,radius=.08,candidates=5,neighbors=4,common_min=3)
    assert not np.any(g['sources'][g['spatial_weights']>0]==4)


def test_confidence_reduces_hidden_key_tether_without_pinning_unseen_3d():
    g=torch_graph();g['key_confidence'][0]=0
    x=torch.tensor(example()[0][0],dtype=torch.float64)+.01
    c=g['reference_covariance'][0].repeat(len(x),1,1)
    a,_=propagate(x,c,g,0,torch.arange(len(g['targets'])))
    b,_=propagate(x,c,g,0,torch.arange(len(g['targets'])),uniform=True)
    assert a['tether']==0 and b['tether']>0


def test_camera_depth_precision_is_weaker_than_image_plane_tether():
    g=torch_graph();x=torch.tensor(example()[0][0],dtype=torch.float64)
    c=g['reference_covariance'][0].repeat(len(x),1,1);rows=torch.arange(len(g['targets']))
    a,_=propagate(x+torch.tensor([.01,0.,0.]),c,g,0,rows)
    b,_=propagate(x+torch.tensor([0.,0.,.01]),c,g,0,rows)
    assert a['tether']>9*b['tether']


def test_covariance_stretch_receives_finite_restoring_gradient():
    g=torch_graph();x=torch.tensor(example()[0][0],dtype=torch.float64)
    c=(g['reference_covariance'][0].repeat(len(x),1,1)*2).requires_grad_(True)
    terms,_=propagate(x,c,g,0,torch.arange(len(g['targets'])))
    terms['covariance'].backward()
    assert c.grad.isfinite().all() and c.grad.diagonal(dim1=-2,dim2=-1).sum()>0


def test_flags_off_is_same_scalar_and_does_not_evaluate_new_branch():
    x=torch.tensor(2.,requires_grad=True)
    def forbidden(): raise AssertionError('disabled branch evaluated')
    assert extend(x,False,forbidden) is x


def test_reference_can_fall_back_from_better_looking_but_geometrically_unsupported_time():
    x,s,q,op,keys,tr=example(); s[0]=10
    x[0,:,:2]=0  # All sources are collinear only in the highest-score frame.
    g=build_patches(x,s,q,op,keys,tr,radius=.08,candidates=5,neighbors=4)
    assert (g['reference_indices']>0).all()


def test_missing_neighbors_have_zero_weight_instead_of_duplicated_constraints():
    g=build_patches(*example(),radius=.08,candidates=8,neighbors=8)
    counts=(g['spatial_weights']>0).sum(1)
    assert (counts<8).all() and (counts>=3).all()
    np.testing.assert_allclose(g['spatial_weights'].sum(1),np.ones(len(counts)),rtol=1e-6)


def test_three_noncollinear_sources_determine_a_proper_rigid_patch():
    x,s,q,op,keys,tr=example();tr[:,3:]=False;keys=keys[:3]
    g=build_patches(x,s,q,op,keys,tr,radius=.08,candidates=8,neighbors=8)
    assert len(g['targets'])>0
    assert ((g['spatial_weights']>0).sum(1)==3).all()
