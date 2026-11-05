import numpy as np
import torch
from variants.staged_reconstruction.i10_local_motion.core import support,add_local


def track(start=0,anchor=0,other=1,visible=(True,False)):
    return dict(start=start,uv=[[2,3],[4,5]],accepted=True,binding=dict(accepted=True,row=anchor),
                reciprocal=dict(passed=True),visibility=dict(passed=True),
                visibility_by_row={str(anchor):[dict(passed=v) for v in visible]},
                neighborhood=dict(qualified=True,rows=[other]))


def test_occluded_and_outside_window_have_no_pseudo_targets():
    xyz=np.zeros((5,2,3),np.float32);xyz[:,1,0]=1
    obs,pairs,s= support(dict(rows=[track(start=1)]),xyz)
    assert list(obs)==list(pairs)==[1] and s['supported_pair_frames']==1


def test_duplicate_uv_mean_and_undirected_pair_deduplication():
    xyz=np.zeros((2,2,3),np.float32);xyz[:,1,0]=1
    t=track();u=track();u['uv'][0]=[4,5]
    obs,pairs,_=support(dict(rows=[t,u]),xyz)
    assert obs[0]==[(0,[3.,4.])] and pairs[0]==[(0,1,[1.,0.,0.])]


def test_rejected_binding_cannot_enter_graph():
    t=track();t['accepted']=False
    obs,pairs,s=support(dict(rows=[t]),np.zeros((2,2,3)))
    assert obs==pairs=={} and s['accepted_tracks']==0


def test_unqualified_graph_retains_verified_reprojection_only():
    t=track();t['neighborhood']['qualified']=False
    obs,pairs,_=support(dict(rows=[t]),np.zeros((2,2,3)))
    assert 0 in obs and not pairs


def test_flags_off_and_empty_return_exact_base_without_access():
    base=torch.tensor(2.,requires_grad=True)
    assert add_local(base,None,None)[0] is base
    assert add_local(base,None,[],True)[0] is base


def test_global_patch_translation_is_unpenalized():
    world=torch.tensor([[1.,2.,3.],[1.1,2.,3.]],requires_grad=True)
    refs=[(0,1,[.1,0,0])]
    _,a=add_local(torch.tensor(0.),world,refs,True)
    _,b=add_local(torch.tensor(0.),world+world.new_tensor([4.,-2.,3.]),refs,True)
    assert a.abs()<1e-10 and b.abs()<1e-10


def test_relative_correction_propagates_finite_balanced_gradient():
    world=torch.tensor([[.01,0.,0.],[.1,0.,0.],[9.,9.,9.]],requires_grad=True)
    _,loss=add_local(torch.tensor(0.),world,[(0,1,[.1,0,0])],True)
    loss.backward()
    assert loss>0 and torch.isfinite(world.grad).all() and world.grad[0].abs().sum()>0
    assert torch.equal(world.grad[0],-world.grad[1]) and world.grad[2].abs().sum()==0


def test_parent_time_varying_deformation_is_not_global_rigidity():
    xyz=np.zeros((2,2,3),np.float32);xyz[0,1,0]=.1;xyz[1,1,0]=.2
    _,pairs,_=support(dict(rows=[track(visible=(True,True))]),xyz)
    for t in range(2):
        _,loss=add_local(torch.tensor(0.),torch.tensor(xyz[t]),pairs[t],True)
        assert loss==0


def test_zero_residual_has_finite_zero_gradient():
    world=torch.tensor([[0.,0,0],[.1,0,0]],requires_grad=True)
    _,loss=add_local(torch.tensor(0.),world,[(0,1,[.1,0,0])],True)
    loss.backward();assert torch.equal(world.grad,torch.zeros_like(world))
