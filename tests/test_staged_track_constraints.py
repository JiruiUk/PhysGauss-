import torch
from variants.staged_reconstruction.stage4.constraints import track_projection,fixed_row_track_loss


def test_disabled_track_loss_does_not_read_scene_or_tracks():
    assert track_projection(None,None) is None
    assert fixed_row_track_loss(None,None,None) is None


def test_pixel_displacement_gradient_reduces_reprojection_error():
    p=torch.tensor([[3.,5.],[6.,7.]],requires_grad=True);target=torch.tensor([[5.,5.],[6.,8.]])
    value=track_projection(p,target,enabled=True);value.backward()
    assert p.grad[0,0]<0 and p.grad[0,1]==0 and p.grad[1,1]<0


def test_unknown_nonfinite_track_target_is_not_filled_or_supervised():
    p=torch.tensor([[3.,5.],[6.,7.]],requires_grad=True);target=torch.tensor([[float('nan'),5.],[6.,8.]])
    value=track_projection(p,target,torch.tensor([0.,1.]),True);value.backward()
    assert torch.equal(p.grad[0],torch.zeros(2)) and p.grad[1,1]<0
