"""Read-only diagnostic subsets and explicit denominator accounting."""
import hashlib
import json
import numpy as np


def select_ids(ids,mask,population):
    assert ids.ndim==3 and tuple(ids.shape[1:])==tuple(mask.shape)
    values=np.asarray(ids)[:,np.asarray(mask,dtype=bool)].reshape(-1)
    return np.unique(values[(values>=0)&(values<population)]).astype(np.int64)


def weighted_observation(weight,rgb):
    weight=np.asarray(weight,dtype=np.float64);rgb=np.asarray(rgb,dtype=np.float64)
    assert weight.shape==rgb.shape[:2] and rgb.shape[2]==3
    assert np.isfinite(weight).all() and (weight>=0).all()
    total=float(weight.sum())
    if total==0:return {'visible_weight':0.,'observed_mean_RGB':None,'weight_on_observed_RGB_lt005':None}
    gray=rgb.mean(2)
    return {'visible_weight':total,'observed_mean_RGB':float((weight*gray).sum()/total),
        'weight_on_observed_RGB_lt005':float(weight[gray<.05].sum()/total)}


def metadata_census(frames):
    times={};cameras={}
    for frame in frames:
        t=float(frame['time']).hex()
        camera=hashlib.sha256(json.dumps(frame['transform_matrix'],separators=(',',':')).encode()).hexdigest()
        times.setdefault(t,[]).append(camera);cameras.setdefault(camera,[]).append(t)
    return {'frames':len(frames),'unique_times':len(times),'unique_camera_transforms':len(cameras),
        'time_camera_counts':[{'time_hex':t,'images':len(v),'unique_camera_transforms':len(set(v))} for t,v in sorted(times.items(),key=lambda x:float.fromhex(x[0]))],
        'same_time_multicamera_times':sum(len(set(v))>1 for v in times.values()),
        'camera_time_counts':sorted(len(set(v)) for v in cameras.values())}


def packing_colors(population,static_count,groups):
    assert len(groups)<=3
    colors=np.zeros((population+static_count,3),dtype=np.float32)
    for channel,ids in enumerate(groups):
        assert np.all((np.asarray(ids)>=0)&(np.asarray(ids)<population))
        colors[static_count+np.asarray(ids,dtype=np.int64),channel]=1
    return colors
