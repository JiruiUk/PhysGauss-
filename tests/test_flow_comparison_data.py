import torch
from variants.flow_comparison.geometry import pixel_grid,sample,project,motion_target,forward_backward_valid,flow_loss


def test_static_camera_object_translation_and_fb_consistency():
    h,w=12,20;f=torch.zeros(h,w,2,dtype=torch.double);f[...,0]=2;b=-f
    pose=torch.eye(4,dtype=torch.double);K=torch.tensor([[10.,0.,10.],[0.,10.,6.],[0.,0.,1.]],dtype=torch.double)
    motion,valid=motion_target(f,torch.full((h,w),4.,dtype=torch.double),pose,K,pose,K)
    mask=valid&forward_backward_valid(f,b)
    torch.testing.assert_close(motion[mask],f[mask]);assert mask.sum()==h*(w-2)
    bad=b.clone();bad[...,0]=8;assert not forward_backward_valid(f,bad).any()


def test_camera_translation_compensation_gives_zero_static_motion():
    h,w=12,20;pose=torch.eye(4,dtype=torch.double);next_pose=pose.clone();next_pose[0,3]=.4
    K=torch.tensor([[10.,0.,10.],[0.,10.,6.],[0.,0.,1.]],dtype=torch.double)
    f=torch.zeros(h,w,2,dtype=torch.double);f[...,0]=1 # fx*translation/depth
    motion,valid=motion_target(f,torch.full((h,w),4.,dtype=torch.double),pose,K,next_pose,K)
    torch.testing.assert_close(motion[valid],torch.zeros_like(motion[valid]),atol=1e-12,rtol=0)


def test_normalization_is_by_component_not_image_row_and_no_saturation():
    target=torch.zeros(12,20,2,dtype=torch.double)
    pred=torch.full_like(target,2.,requires_grad=True);valid=torch.ones(12,20,dtype=torch.double)
    loss=flow_loss(pred,target,valid);assert abs(loss.item()-(2/20+2/12)/2)<1e-12
    loss.backward();assert (pred.grad>0).all()
    assert torch.equal(target,torch.zeros_like(target))


def test_zero_valid_pixels_is_finite_and_differentiable():
    pred=torch.randn(8,10,2,requires_grad=True);loss=flow_loss(pred,torch.zeros_like(pred),torch.zeros(8,10))
    loss.backward();assert loss.item()==0 and torch.isfinite(pred.grad).all()


def test_invalid_target_nan_is_excluded():
    target=torch.zeros(8,10,2);target[0]=float('nan');mask=torch.ones(8,10);mask[0]=0
    pred=torch.ones_like(target,requires_grad=True);loss=flow_loss(pred,target,mask)
    loss.backward();assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
