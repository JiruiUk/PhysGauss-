from types import SimpleNamespace
import torch
from variants.staged_reconstruction.recovery.coverage import chamfer,coverage_loss


def test_disjoint_silhouette_points_receive_toward_target_gradient():
    p=torch.tensor([[10.,10.],[12.,10.]],requires_grad=True);q=torch.tensor([[40.,10.],[42.,10.]])
    loss=chamfer(p,q,100);loss.backward()
    assert loss>0 and (p.grad[:,0]<0).all() and p.grad[:,1].abs().max()<1e-6


def test_unknown_union_is_invariant_to_unknown_model_id_permutation():
    camera=SimpleNamespace(full_proj_transform=torch.eye(4),image_width=100,image_height=100)
    p=torch.tensor([[-.5,0.,1.],[.5,0.,1.]],requires_grad=True);labels=torch.full((100,100),-1)
    labels[45:55,20:30]=0;labels[45:55,70:80]=1
    a=coverage_loss(p,torch.tensor([0,1]),camera,labels,[-1,-1]);b=coverage_loss(p,torch.tensor([1,0]),camera,labels,[-1,-1])
    assert torch.allclose(a,b)
    a.backward();assert p.grad.isfinite().all()


def test_known_group_cannot_choose_another_groups_component():
    camera=SimpleNamespace(full_proj_transform=torch.eye(4),image_width=100,image_height=100)
    p=torch.tensor([[-.5,0.,1.],[.5,0.,1.]])
    labels=torch.full((100,100),-1);labels[49:51,24:26]=0;labels[49:51,74:76]=1
    correct=coverage_loss(p,torch.tensor([0,1]),camera,labels,[0,1]);wrong=coverage_loss(p,torch.tensor([1,0]),camera,labels,[0,1])
    assert wrong>correct+.1


def test_empty_unknown_target_does_not_invent_supervision():
    camera=SimpleNamespace(full_proj_transform=torch.eye(4),image_width=100,image_height=100)
    p=torch.tensor([[0.,0.,1.]],requires_grad=True)
    loss=coverage_loss(p,torch.tensor([0]),camera,torch.full((100,100),-1),[-1]);loss.backward()
    assert loss==0 and torch.equal(p.grad,torch.zeros_like(p))
