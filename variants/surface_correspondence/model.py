from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    enabled: bool=False
    min_mass: float=.02
    min_purity: float=.95
    max_flow_error: float=2.
    min_particles: int=20
    min_cohort_mass: float=.2
    min_center_fraction: float=.9
    min_flow_fraction: float=.9


def particle_support(own_mass,total_mass,center_inside,flow_error=None,config=Config()):
    """Actual visible footprint and projected center must agree; no proximity IDs."""
    mass=np.asarray(own_mass);total=np.asarray(total_mass)
    good=np.isfinite(mass)&np.isfinite(total)&(mass>=config.min_mass)&(total>0)
    good&=(mass/np.maximum(total,1e-20)>=config.min_purity)&np.asarray(center_inside,dtype=bool)
    if flow_error is not None:
        good&=np.isfinite(flow_error)&(np.asarray(flow_error)<=config.max_flow_error)
    return good


def freeze_bank(supports,canonical,config=Config()):
    if not config.enabled:return None
    if len(supports)!=3:raise ValueError('Exactly three seed observations required')
    rows=np.flatnonzero(np.logical_and.reduce(supports))
    if len(rows)<config.min_particles:return None
    return {'kernel_ids':rows,'canonical':np.asarray(canonical)[rows].copy()}


def assess_cohort(own,total,center_inside,flow_error,reference_visible,config=Config()):
    own=np.asarray(own);total=np.asarray(total);mass=float(own.sum());image=float(total.sum())
    precision=mass/image if image>1e-12 else None
    center=float(np.sum(total*np.asarray(center_inside))/image) if image>1e-12 else None
    flow=float(np.sum(own*(np.isfinite(flow_error)&(np.asarray(flow_error)<=config.max_flow_error)))/mass) if mass>1e-12 else None
    reasons=[]
    if not np.isfinite(own).all() or not np.isfinite(total).all():reasons.append('nonfinite_contribution')
    if not reference_visible:reasons.append('image_fragment_unverified_or_terminated')
    if mass<config.min_cohort_mass:reasons.append('insufficient_actual_visible_footprint')
    if precision is None or precision<config.min_purity:reasons.append('footprint_outside_fragment')
    if center is None or center<config.min_center_fraction:reasons.append('projected_centers_not_supported')
    if flow is None or flow<config.min_flow_fraction:reasons.append('center_motion_disagrees_or_unobserved')
    return {'accepted':config.enabled and not reasons,'unknown_reasons':reasons,'own_mass':mass,'all_image_mass':image,
            'footprint_purity':precision,'mass_weighted_center_inside':center,'own_mass_weighted_flow_agreement':flow}


def optional_bank(supports=None,canonical=None,config=Config()):
    if not config.enabled:return None
    return freeze_bank(supports,canonical,config)


def qualify_fragments(labels,components,alive):
    """Conservative visible component links; a terminated local seed never revives."""
    mapping=np.full(4,-1,dtype=np.int16);alive=np.asarray(alive,dtype=bool).copy()
    for k in range(int(np.max(components))+1):
        region=components==k;known=labels[region];known=known[known>=0]
        if len(known)<30 or len(known)<.25*region.sum():continue
        count=np.bincount(known,minlength=4);winner=int(count.argmax())
        if count[winner]/len(known)<.95 or not alive[winner]:continue
        if mapping[winner]>=0:mapping[winner]=-2
        elif mapping[winner]!=-2:mapping[winner]=k
    alive&=mapping>=0;mapping[~alive]=-1
    return mapping,alive
