"""Motion-compatible elastic neighborhoods and renderer-exact covariance curvature."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def motion_gate(canonical,tracks,anchors,neighbors,valid,absolute=.02,relative=1.5):
    if len(tracks)<2:raise ValueError('Need multiple motion observations')
    displacement=torch.stack(tracks)-tracks[0][None]
    contrast=(displacement[:,neighbors]-displacement[:,anchors,None]).norm(dim=-1).amax(0)
    distance=(canonical[neighbors]-canonical[anchors,None]).norm(dim=-1)
    return valid&(contrast<=absolute+relative*distance)


def rendered_covariance(scale,raw_quaternion):
    """Match CUDA GLM column ordering and its non-normalized quaternion exactly.

    The GLM R in forward.cu is the transpose of the row-major matrix below;
    CUDA computes (S R)^T(S R), hence this expression is R_row S^2 R_row^T.
    No normalization or canonical scale/sign convention is changed.
    """
    r,x,y,z=raw_quaternion.unbind(-1)
    matrix=torch.stack((1-2*(y*y+z*z),2*(x*y-r*z),2*(x*z+r*y),
        2*(x*y+r*z),1-2*(x*x+z*z),2*(y*z-r*x),
        2*(x*z-r*y),2*(y*z+r*x),1-2*(x*x+y*y)),-1).reshape(*r.shape,3,3)
    transformed=matrix*scale[...,None,:]
    return transformed@transformed.transpose(-1,-2)


def packed_covariance(matrix):
    return torch.stack([matrix[...,i,j] for i,j in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]],-1)


def covariance_curvature(a,b,c,shape_tolerance=.005,size_tolerance=.05):
    def decompose(cov):
        trace=cov.diagonal(dim1=-2,dim2=-1).sum(-1).clamp_min(1e-12)
        return cov/trace[...,None,None],trace.log()
    sa,za=decompose(a);sb,zb=decompose(b);sc,zc=decompose(c)
    shape=((sa-2*sb+sc).square().sum((-2,-1))+1e-16).sqrt()/3
    size=(za-2*zb+zc).abs()
    es=F.relu(shape-shape_tolerance);ez=F.relu(size-size_tolerance)
    loss=F.smooth_l1_loss(es,torch.zeros_like(es),beta=.02)+.1*F.smooth_l1_loss(ez,torch.zeros_like(ez),beta=.1)
    return loss,{'shape_curvature':shape.detach().mean(),'log_trace_curvature':size.detach().mean()}


def time_stencil(t,h):
    if not 0<=t<=1 or not 0<h<.25:raise ValueError('Invalid normalized time stencil')
    if t<h:return t,t+h,t+2*h
    if t>1-h:return t-2*h,t-h,t
    return t-h,t,t+h
