"""Source-view depth envelopes; missing evidence never supplies a negative vote."""
import numpy as np
from ..adaptive_static.prepare import world_points
from output.short_clip_diagnostic.camera_math import project


def pose_ids(views,decimals=9):
    rounded = np.round(np.asarray(views),decimals).astype('<f8')
    rounded[rounded == 0] = 0
    keys = [v.tobytes() for v in rounded]
    identities = {}; return np.asarray([identities.setdefault(k,len(identities)) for k in keys])


def observations(uv,z,views,k,index,sources,depths,relative=.025):
    xyz = world_points(uv,z,views[index],k); count = len(z)
    returned = np.zeros((count,len(sources)),np.float64)
    error = returned.copy(); available = np.zeros_like(returned,dtype=bool); closer = available.copy()
    for col,j in enumerate(sources):
        pixels,expected = project(xyz,views[j],k); height,width = depths[j].shape
        inside = (expected > .2)&np.isfinite(pixels).all(1)&(pixels[:,0] >= 1)&(pixels[:,0] < width-1)&(pixels[:,1] >= 1)&(pixels[:,1] < height-1)
        ids = np.flatnonzero(inside); xy = np.rint(pixels[ids]).astype(int)
        sampled = depths[j][xy[:,1],xy[:,0]]
        good = np.isfinite(sampled)&(sampled > 0); ids,xy,sampled = ids[good],xy[good],sampled[good]
        world = world_points(xy,sampled,views[j],k)
        back,back_z = project(world,views[index],k)
        good = np.isfinite(back).all(1)&np.isfinite(back_z)&(back_z > .2)
        ids,back,back_z,sampled = ids[good],back[good],back_z[good],sampled[good]
        returned[ids,col] = back_z; error[ids,col] = np.linalg.norm(back-uv[ids],axis=1)
        available[ids,col] = True; closer[ids,col] = sampled < expected[ids]*(1-relative)
    return returned,available,error,closer


def envelope(z,returned,available,reprojection,closer,source_pose_ids,source_times,reference_time,near,far,protocol):
    z = np.asarray(z); assert np.isfinite(z).all() and (z > 0).all()
    available = available&np.isfinite(returned)&(returned > 0)&np.isfinite(reprojection)
    step = abs((1/far-1/near)/(protocol['depth_samples']-1))
    inv = 1/z
    assert np.all(inv > step)
    spacing = np.maximum(abs(1/(inv-step)-z),abs(1/(inv+step)-z))
    lo = np.minimum(z,np.where(available,returned,np.inf).min(1))-.5*spacing
    hi = np.maximum(z,np.where(available,returned,-np.inf).max(1))+.5*spacing
    width = hi-lo; counts = available.sum(1)
    identities = np.unique(source_pose_ids)
    distinct = np.stack([(available[:,np.asarray(source_pose_ids) == key]).any(1) for key in identities],1).sum(1)
    first = np.minimum(reference_time,np.where(available,source_times[None],np.inf).min(1))
    last = np.maximum(reference_time,np.where(available,source_times[None],-np.inf).max(1))
    span = last-first
    maximum = np.where(available,reprojection,0).max(1)
    median = np.zeros(len(z))
    for i in np.flatnonzero(counts): median[i] = np.median(reprojection[i,available[i]])
    score = distinct/max(len(identities),1)*np.minimum(1,span/protocol['minimum_time_span_frame_indices'])
    score *= np.exp(-median/protocol['maximum_reprojection_px'])/(1+width/spacing)
    tight = (distinct >= protocol['minimum_distinct_source_poses'])&(span >= protocol['minimum_time_span_frame_indices'])
    tight &= (maximum <= protocol['maximum_reprojection_px'])&(width <= protocol['maximum_tight_interval_width'])
    if protocol['tight_requires_no_closer_surface_ambiguity']: tight &= ~(closer&available).any(1)
    return {'lower':lo,'upper':hi,'width':width,'confidence_score':score,'tight':tight,
        'available_sources':counts,'distinct_source_poses':distinct,'source_time_span':span,
        'maximum_reprojection_px':maximum,'closer_surface_unknown_count':(closer&available).sum(1),
        'source_returned_depth':returned,'source_available':available,'source_reprojection_px':reprojection}


def add_evidence(z,uv=None,views=None,k=None,index=None,sources=None,depths=None,frame_ids=None,near=None,far=None,protocol=None,enabled=False):
    if not enabled: return z,None
    assert len(sources) == 6
    source = observations(uv,z,views,k,index,sources,depths,protocol['closer_surface_relative_tolerance'])
    identities = pose_ids(views,protocol['camera_pose_round_decimals'])[sources]
    result = envelope(z,*source,identities,np.asarray(frame_ids)[sources],frame_ids[index],near,far,protocol)
    assert all(np.isfinite(v).all() for v in result.values())
    assert np.all((result['confidence_score'] >= 0)&(result['confidence_score'] <= 1))
    return z,result
