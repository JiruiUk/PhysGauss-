"""Image coverage and tolerant short-term neighborhood strain, without instance IDs."""
import math
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from variants.coverage_tuning.model import interior


def coverage_objective(alpha,target,goal=.95,background_weight=.25):
    if alpha.shape!=target.shape:raise ValueError('Alpha/target shape mismatch')
    core=interior(target.detach());bg=target.detach()<=.5
    deficit=(F.relu(goal-alpha)*core).sum()/core.sum().clamp_min(1)
    excess=(alpha*bg).sum()/bg.sum().clamp_min(1)
    return deficit+background_weight*excess


def strain_objective(now,later,anchors,neighbors,valid,tolerance=.05):
    a=torch.linalg.vector_norm(now[neighbors]-now[anchors,None],dim=-1)
    b=torch.linalg.vector_norm(later[neighbors]-later[anchors,None],dim=-1)
    # Short-term, rotation invariant, tolerant of real elastic strain.
    error=F.relu((b.clamp_min(1e-5).log()-a.clamp_min(1e-5).log()).abs()-math.log1p(tolerance))
    penalty=F.smooth_l1_loss(error,torch.zeros_like(error),beta=.1,reduction='none')
    return (penalty*valid).sum()/valid.sum().clamp_min(1)


@torch.no_grad()
def persistent_edges(positions,anchors,neighbors,valid,max_distance=.15):
    result=valid.clone()
    for point in positions:
        distance=torch.linalg.vector_norm(point[neighbors]-point[anchors,None],dim=-1)
        result &= (distance<max_distance)&(distance>1e-5)
    return result


def image_metrics(alpha,target,rgb,gt):
    """2D silhouette/appearance proxies, not 3D watertightness or instance accuracy."""
    a=np.asarray(alpha,dtype=np.float32).squeeze();m=np.asarray(target,dtype=np.float32).squeeze()
    fg=m>.5;bg=~fg;pred=a>.5
    core=ndimage.minimum_filter(m,size=7,mode='constant',cval=0)>.95
    error=np.abs(np.asarray(rgb)-np.asarray(gt)).mean(-1)
    target_labels,count=ndimage.label(fg)
    fragmented=[];extras=[]
    for label in range(1,count+1):
        component=target_labels==label;area=int(component.sum())
        if area<32:continue
        parts,n=ndimage.label(component&pred)
        sizes=np.bincount(parts.ravel())[1:];sizes=sizes[sizes>=max(8,int(.001*area))]
        fragmented.append(1.-(int(sizes.max()) if len(sizes) else 0)/area)
        extras.append(max(0,len(sizes)-1))
    def rate(mask,domain):return float((mask&domain).sum()/max(1,int(domain.sum())))
    record={'core_low_50':rate(a<.5,core),'core_low_90':rate(a<.9,core),
        'foreground_missing_50':rate(~pred,fg),'background_false_coverage':rate(pred,bg),
        'fragment_loss':float(np.mean(fragmented)) if fragmented else 0.,
        'extra_fragments':float(np.mean(extras)) if extras else 0.,
        'opaque_appearance_error':rate((a>=.9)&(error>.15),fg),
        'foreground_RGB_L1':float(error[fg].mean()) if fg.any() else 0.,
        'mask_iou':float((fg&pred).sum()/max(1,int((fg|pred).sum())))}
    record['integrity_score']=.45*record['core_low_90']+.25*record['foreground_missing_50']+\
        .2*record['fragment_loss']+.1*record['opaque_appearance_error']+.1*record['background_false_coverage']
    return record


def aggregate(rows):
    values={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key not in ('frame','index')}
    values['worst_score']=max(r['integrity_score'] for r in rows)
    values['p95_score']=float(np.quantile([r['integrity_score'] for r in rows],.95))
    return values
