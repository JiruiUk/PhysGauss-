"""Fixed, explicitly labelled neighborhoods; no learned segmentation or material law."""
import numpy as np
from scipy import ndimage
import torch


def clear_duck_masks(rgba, minimum_area=20):
    """Scene-specific image candidates. Must be visually reviewed, never GT kernel IDs."""
    c = np.asarray(rgba, dtype=np.float32)/255
    rgb, alpha = c[..., :3], c[..., 3]
    r,g,b = rgb.transpose(2,0,1)
    yellow = (r>.25)&(g>.18)&(b<.8*np.minimum(r,g))&(alpha>.95)
    yellow = ndimage.binary_fill_holes(ndimage.binary_closing(yellow, iterations=1))
    labels, count = ndimage.label(yellow)
    groups=[]
    for k in range(1,count+1):
        mask=labels==k
        if mask.sum()>=minimum_area:
            groups.append((np.argwhere(mask)[:,0].mean(),mask))
    groups.sort(key=lambda item:item[0])
    # Four separated ducks are verified in early training images of this scene.
    if len(groups)!=4:
        raise ValueError(f'Expected four visually separated training candidates; got {len(groups)}')
    result=np.full(alpha.shape,-1,dtype=np.int64)
    for i,(_,mask) in enumerate(groups):
        result[ndimage.binary_erosion(mask,iterations=1)]=i
    return result


@torch.no_grad()
def fixed_graph(reference, labels, anchors=256, k=12, radius=.08):
    """Reference is frozen; unknown IDs never connect, even to another unknown."""
    aa=[];nn=[];valid=[]
    for group in sorted(labels[labels>=0].unique().tolist()):
        ids=torch.where(labels==group)[0]
        if len(ids)<k+1:continue
        count=min(anchors,len(ids))
        chosen=ids[torch.arange(count,device=ids.device)*len(ids)//count]
        for chunk in chosen.split(32):
            dist=torch.cdist(reference[chunk],reference[ids])
            dist.masked_fill_(chunk[:,None]==ids[None],float('inf'))
            values,index=dist.topk(k,largest=False)
            aa.append(chunk);nn.append(ids[index]);valid.append(values<radius)
    if not aa:
        raise ValueError('No labelled, sufficiently supported shape neighborhoods')
    a,n,v=torch.cat(aa),torch.cat(nn),torch.cat(valid)
    dx=(reference[n]-reference[a,None])*v[...,None]
    sv=torch.linalg.svdvals(dx.double())
    usable=(v.sum(-1)>=6)&(sv[:,-1]>1e-3*sv[:,0].clamp_min(1e-10))
    if not usable.any():raise ValueError('All shape neighborhoods are geometrically degenerate')
    return a[usable],n[usable],v[usable]


def local_shape(reference, positions, anchors, neighbors, valid, lower=.3, upper=3.):
    """A differentiable local F estimate plus optional affine and collapse/stretch terms.

    F is relative to a clear reference pose, not canonical physical strain. Only
    positions receive gradients. Surface-point neighborhoods are rank-gated.
    """
    if not 0<lower<1<upper:raise ValueError('Invalid singular-value bounds')
    dx=(reference.detach()[neighbors]-reference.detach()[anchors,None]).double()
    dy=(positions[neighbors]-positions[anchors,None]).double()
    w=valid[...,None].to(dx)
    dx,dy=dx*w,dy*w
    with torch.no_grad():
        basis=torch.linalg.pinv(dx,rtol=1e-4)
        scale=dx.square().sum((-1,-2)).clamp_min(1e-10)
    mapping=basis@dy
    residual=((dx@mapping-dy).square()*w).sum((-1,-2))/scale
    singular=torch.linalg.svdvals(mapping)
    stretch=(torch.relu(lower-singular).square()+torch.relu(singular-upper).square()).mean(-1)
    orientation=torch.relu(.02-torch.linalg.det(mapping)).square()
    return {'affine':residual.mean().to(positions.dtype),
            'collapse':(stretch+orientation).mean().to(positions.dtype),
            'F':mapping.transpose(-1,-2),'singular':singular,
            'det':torch.linalg.det(mapping),'residual_per_anchor':residual}
