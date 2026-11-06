import torch
from torch import nn


def heteroscedastic_flow_loss(predicted_flow, target_flow, log_variance, valid_mask=None):
    assert predicted_flow.shape == target_flow.shape and predicted_flow.shape[-1] == 2
    if log_variance.shape == predicted_flow.shape[:-1]: log_variance = log_variance.unsqueeze(-1)
    if log_variance.shape not in (predicted_flow.shape, predicted_flow.shape[:-1]+(1,)):
        raise ValueError("Flow log variance must broadcast only across the two vector components")
    finite = torch.isfinite(predicted_flow).all(-1) & torch.isfinite(target_flow).all(-1)
    finite = finite & torch.isfinite(log_variance).all(-1)
    if valid_mask is not None:
        assert valid_mask.shape == finite.shape
        finite = finite & valid_mask.bool()
    # Select before arithmetic: masked NaNs must never contaminate derivatives.
    if not finite.any():
        return predicted_flow[finite].sum()+log_variance[finite].sum()
    error = (predicted_flow[finite]-target_flow[finite]).square()
    s = log_variance[finite]
    return (torch.exp(-s)*error+s).mean()


def flow_valid_mask(prediction, observation, occlusion=None, projection_valid=None):
    H,W,_ = prediction.shape
    y,x = torch.meshgrid(torch.arange(H,device=prediction.device),torch.arange(W,device=prediction.device),indexing="ij")
    endpoint = observation + torch.stack([x,y],-1)
    valid = (torch.isfinite(prediction).all(-1) & torch.isfinite(observation).all(-1) &
             (endpoint[...,0]>=0) & (endpoint[...,0]<=W-1) &
             (endpoint[...,1]>=0) & (endpoint[...,1]<=H-1))
    predicted_endpoint = prediction+torch.stack([x,y],-1)
    valid = valid & (predicted_endpoint[...,0]>=0) & (predicted_endpoint[...,0]<=W-1)
    valid = valid & (predicted_endpoint[...,1]>=0) & (predicted_endpoint[...,1]<=H-1)
    if occlusion is not None:
        assert occlusion.shape == (H,W)
        valid = valid & ~occlusion.bool()
    if projection_valid is not None:
        assert projection_valid.shape == (H,W)
        valid = valid & projection_valid.bool()
    return valid


class FlowUncertainty(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,8),nn.Tanh(),nn.Linear(8,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, flow):
        return 6*torch.tanh(self.net(flow.detach()/10)/6)
