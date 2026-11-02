"""Guard shared physics gradients while preserving private stress gradients.

This is an explicit first-order gradient policy, not a rescaled scalar objective.
The surrogate has the true forward residual value, with tested per-parameter
gradients. No higher-order derivative of the gradient policy is intended.
"""
import math
import torch


def balance(image, physical, groups, private, cap, weight):
    if not math.isfinite(cap) or not 0 < cap <= 1 or not math.isfinite(weight) or weight <= 0:
        raise ValueError('Invalid gradient guard ratio or physics weight')
    groups={k:[p for p in ps if p.requires_grad] for k,ps in groups.items()}
    shared=[p for ps in groups.values() for p in ps]
    private=[p for p in private if p.requires_grad]
    params=shared+private
    if not params or len({id(p) for p in params}) != len(params):
        raise ValueError('Guard requires disjoint parameter groups')
    gi=torch.autograd.grad(image,shared,allow_unused=True,retain_graph=True)
    gp=torch.autograd.grad(physical,params,allow_unused=True,retain_graph=True)
    if any(not bool(g.detach().isfinite().all()) for g in gp if g is not None):
        raise FloatingPointError('Nonfinite private or shared momentum gradient')
    rows={}; offset=0; scale=1.
    def norm(gs):
        values=[g.detach().float().square().sum() for g in gs if g is not None]
        value=float(torch.stack(values).sum().sqrt()) if values else 0.
        if not math.isfinite(value): raise FloatingPointError('Nonfinite guard gradient')
        return value
    for key,ps in groups.items():
        n=len(ps); ni=norm(gi[offset:offset+n]); np=weight*norm(gp[offset:offset+n]);offset+=n
        if np > 0: scale=min(scale,cap*ni/np)
        rows[key]={'image_norm':ni,'weighted_momentum_norm':np}
    if not bool(physical.detach().isfinite()): raise FloatingPointError('Nonfinite momentum residual')
    # Preserve exact existing autograd path if no guard is needed.
    if scale==1:
        result=physical
    else:
        result=physical.detach()
        for index,(p,g) in enumerate(zip(params,gp)):
            if g is None: continue
            multiplier=scale if index<len(shared) else 1.
            result=result+((p-p.detach())*(g.detach()*multiplier)).sum()
    for row in rows.values(): row['guarded_momentum_norm']=scale*row['weighted_momentum_norm']
    return result,{'scale':scale,'cap':cap,'groups':rows,'private_gradients_preserved':True}
