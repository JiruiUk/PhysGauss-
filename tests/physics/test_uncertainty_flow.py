import torch
from losses.uncertainty_flow import heteroscedastic_flow_loss, flow_valid_mask


def test_likelihood_masks_and_gradients():
    prediction=torch.zeros(3,4,2,requires_grad=True)
    target=torch.ones_like(prediction)
    variance=torch.zeros(3,4,1,requires_grad=True)
    mask=torch.ones(3,4,dtype=torch.bool);mask[0,0]=False
    loss=heteroscedastic_flow_loss(prediction,target,variance,mask)
    torch.testing.assert_close(loss,torch.tensor(1.))
    loss.backward();assert prediction.grad[0,0].abs().sum()==0
    occlusion=torch.zeros(3,4,dtype=torch.bool);occlusion[1,1]=True
    valid=flow_valid_mask(prediction.detach(),target,occlusion,torch.ones_like(occlusion))
    assert not valid[1,1] and not valid[-1].any() and not valid[:,-1].any()
    empty=heteroscedastic_flow_loss(prediction,target,variance,mask&False)
    assert empty==0
