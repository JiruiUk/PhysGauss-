"""Read-only support diagnostics. Unknown labels never acquire a nearest ID."""
import numpy as np
from scipy import ndimage
import torch
from variants.flow_comparison.geometry import pixel_grid, sample, forward_backward_valid


def components(rgba, minimum_area=20):
    c=np.asarray(rgba,dtype=np.float32)/255
    r,g,b=c[...,:3].transpose(2,0,1)
    mask=(c[...,3]>.95)&(r>.25)&(g>.18)&(b<.8*np.minimum(r,g))
    mask=ndimage.binary_fill_holes(ndimage.binary_closing(mask,iterations=1))
    labels,count=ndimage.label(mask); groups=[]
    for k in range(1,count+1):
        a=labels==k
        if a.sum()>=minimum_area:groups.append(a)
    groups.sort(key=lambda a:np.argwhere(a)[:,0].mean())
    result=np.full(mask.shape,-1,dtype=np.int64)
    for k,a in enumerate(groups):result[a]=k
    return result


def propagate(labels, target_components, forward, backward):
    """Visible-only FB-consistent advection; merged identities become unknown.

    This is estimated 2D evidence, not material identity or occlusion re-ID.
    Nothing initializes new labels after the initial frame.
    """
    labels=torch.as_tensor(labels,device=forward.device)
    tc=torch.as_tensor(target_components,device=forward.device)
    dest=pixel_grid(*backward.shape[:2],backward)+backward
    h,w=labels.shape
    xy=dest.round().long();inside=(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h)
    moved=labels[xy[...,1].clamp(0,h-1),xy[...,0].clamp(0,w-1)]
    valid=inside&forward_backward_valid(backward,forward)&(sample(forward_backward_valid(forward,backward).float(),dest)>.99)&(tc>=0)
    moved=torch.where(valid,moved,-1)
    ambiguous=[]
    for k in tc[tc>=0].unique().tolist():
        ids=moved[tc==k]; ids=ids[ids>=0].unique()
        if len(ids)>1:
            moved[tc==k]=-1;ambiguous.append(k)
    return moved.cpu().numpy(),ambiguous


class RenderProxy:
    """Override renderer getters without writing model parameters or buffers."""
    def __init__(self, model, **overrides):self.model=model;self.overrides=overrides
    def __getattr__(self,key):
        if key in self.overrides:return self.overrides[key]
        return getattr(self.model,key)


def ratio(a,b):return float(a/b) if float(b)>1e-12 else None


def full_composite(alpha,colors):
    """Tiny independent front-to-back oracle for truncation tests."""
    alpha=np.asarray(alpha); weights=alpha*np.cumprod(np.r_[1.,1-alpha[:-1]])
    return weights@np.asarray(colors),weights
