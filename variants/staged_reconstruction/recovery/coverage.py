"""Image-space coverage objective; unknown regions remain an unlabeled union."""
import torch


def project(positions, camera):
    h=torch.cat([positions,torch.ones_like(positions[:,:1])],-1)@camera.full_proj_transform
    uv=((h[:,:2]/h[:,3:].clamp_min(1e-8)+1)*positions.new_tensor([camera.image_width,camera.image_height])-1)/2
    return uv,h[:,3]>0


def chamfer(predicted, target, width, two_way=True):
    """Symmetric distances support displaced splats even before render overlap.

    Nearest-pixel geometry does not assign/revive a physical object identity.
    Targets have no learned/material labels. The output is dimensionless.
    """
    if not len(predicted) or not len(target):return predicted.sum()*0
    p=predicted/width;t=target/width
    distance=torch.cdist(p[None],t[None]).squeeze(0).square()
    return .5*(distance.min(0).values.mean()+distance.min(1).values.mean()) if two_way else distance.min(0).values.mean()


def sample_rows(values,maximum):
    if len(values)<=maximum:return values
    return values[torch.arange(maximum,device=values.device)*len(values)//maximum]


def coverage_loss(positions, ids, camera, labels, early_owner_to_component):
    """Only qualified IDs get labeled coverage; all remaining objects stay unknown."""
    uv,valid=project(positions,camera)
    labels=labels.to(positions.device)
    terms=[];assigned=[];unknown=[]
    for k,component in enumerate(early_owner_to_component):
        points=uv[(ids==k)&valid]
        if component<0:unknown.append(points);continue
        assigned.append(component)
        yx=torch.nonzero(labels==component);target=yx[:,[1,0]].to(positions)*camera.image_width/labels.shape[1]
        # Observed pixels require support. Hidden surfaces are not forced into
        # the visible mask; RGB handles excess visible splats separately.
        terms.append(chamfer(sample_rows(points,256),sample_rows(target,256),camera.image_width,two_way=False))
    mask=labels>=0
    for k in assigned:mask &= labels!=k
    if unknown:
        yx=torch.nonzero(mask);target=yx[:,[1,0]].to(positions)*camera.image_width/labels.shape[1]
        terms.append(chamfer(sample_rows(torch.cat(unknown),768),sample_rows(target,768),camera.image_width,two_way=False))
    return torch.stack(terms).mean() if terms else positions.sum()*0
