import torch
from variants.staged_reconstruction.stage5.nonpenetration import sample_sdf,nonpenetration


def test_disabled_nonpenetration_does_not_access_inputs():
    assert nonpenetration() is None


def test_only_eligible_deep_penetration_gets_outward_gradient():
    d=torch.tensor([-.2,.3,-.4,-.02],requires_grad=True)
    loss,info=nonpenetration(d,torch.tensor([True,True,False,True]),margin=.05,enabled=True);loss.backward()
    assert d.grad[0]<0 and torch.equal(d.grad[1:],torch.zeros(3))
    assert info['active_violations']==1 and info['eligible_count']==3


def test_no_eligible_observations_is_explicitly_not_validated_contact():
    d=torch.tensor([-.2],requires_grad=True)
    loss,info=nonpenetration(d,torch.tensor([False]),enabled=True);loss.backward()
    assert loss==0 and d.grad==0 and info['no_eligible_samples_is_not_collision_validation']


def test_xyz_volume_axis_contract_and_out_of_grid_queries():
    x=torch.linspace(-1,1,9);field=x[:,None,None].expand(9,7,5).contiguous()
    p=torch.tensor([[-.25,-.5,-.75],[.25,-.5,-.75],[5.,0.,0.]],requires_grad=True)
    d=sample_sdf(p,field,[-1.,-1.,-1.],.25)
    assert torch.allclose(d[:2],torch.tensor([-.25,.25])) and d[2]>0
    d[:2].sum().backward();assert torch.allclose(p.grad[:2],torch.tensor([[1.,0.,0.],[1.,0.,0.]]))
