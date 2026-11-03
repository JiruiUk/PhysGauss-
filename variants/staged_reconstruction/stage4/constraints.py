"""Fixed-row projection supervision from qualified image tracklets, not 3D GT."""
import torch
import torch.nn.functional as F
from ..recovery.coverage import project


def track_projection(predicted,target,confidence=None,enabled=False,width=400):
    if not enabled:return None
    if predicted.shape!=target.shape or predicted.shape[-1]!=2:raise ValueError('Expected aligned image coordinates')
    if confidence is None:confidence=predicted.new_ones(predicted.shape[:-1])
    good=(confidence>0)&torch.isfinite(target).all(-1)
    if not bool(torch.isfinite(predicted[good]).all()):raise ValueError('Nonfinite qualified trajectory projection')
    if not bool(good.any()):return predicted[good].sum()*0
    error=F.huber_loss(predicted[good],target[good],reduction='none',delta=1.).mean(-1)/width
    return (error*confidence[good]).sum()/confidence[good].sum()


def fixed_row_track_loss(scene,batch,cameras,enabled=False):
    if not enabled:return None
    from variants.flow_comparison.temporal import position
    losses=[]
    for item in batch:
        k=item['local_early_object'];row=item['local_row'];obj=scene.objects[k]
        if not 0<=row<len(obj.get_xyz):raise ValueError('Track row invalid after topology change')
        for number,target in zip(item['frames'],item['uv_400']):
            x=position(scene.deforms[k],obj.get_xyz[row:row+1],(number-1)/199)+scene.centers[k]
            uv,valid=project(x,cameras[number])
            if not bool(valid.all()):raise ValueError('Qualified track particle went behind camera')
            losses.append(track_projection(uv,uv.new_tensor([target]),uv.new_tensor([item['confidence']]),True))
    return torch.stack(losses).mean() if losses else scene.centers.sum()*0
