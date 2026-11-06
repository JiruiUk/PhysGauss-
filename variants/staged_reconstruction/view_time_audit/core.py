"""Pose accounting and descriptive render proxies, never geometry truth."""
import numpy as np
import cv2
from variants.view_completeness.metrics import shape_proxies

def pairs(config):
    selected={(c,t) for t in config['anchors'] for c in config['all_camera_indices']}
    n=len(config['all_camera_indices'])
    for anchor in config['anchors']:
        ids=[anchor+d for d in config['offsets']]
        assert all(0<=i<n for i in ids)
        selected.update((c,t) for c in ids for t in ids)
    return sorted(selected,key=lambda x:(x[1],x[0]))

def pose_distance(first,second):
    a,b=np.asarray(first,np.float64),np.asarray(second,np.float64)
    assert a.shape==b.shape==(4,4) and np.isfinite(a).all() and np.isfinite(b).all()
    # Metadata matrices contain float32 rounding; normalize direction and project
    # each rotation to SO(3) before computing its geodesic distance.
    def rotation(m):
        u,_,v=np.linalg.svd(m[:3,:3]);q=u@v
        assert np.linalg.det(q)>0
        return q
    ra,rb=rotation(a),rotation(b)
    cosine=float(np.clip((np.trace(ra.T@rb)-1)/2,-1,1))
    direction=float(np.clip(np.dot(ra[:,2],rb[:,2]),-1,1))
    return dict(rotation_deg=float(np.degrees(np.arccos(cosine))),
        view_direction_deg=float(np.degrees(np.arccos(direction))),
        camera_translation=float(np.linalg.norm(a[:3,3]-b[:3,3])))

def describe(alpha=None,rgb=None,depth=None,config=None,*,enabled=False):
    if not enabled:return None
    alpha=np.asarray(alpha,np.float64);rgb=np.asarray(rgb,np.float64);depth=np.asarray(depth,np.float64)
    assert rgb.shape==alpha.shape+(3,) and depth.shape==alpha.shape
    assert np.isfinite(alpha).all() and np.isfinite(rgb).all() and np.isfinite(depth).all()
    a=np.clip(alpha,0,1);mask=a>config['alpha_threshold'];area=int(mask.sum())
    dark=mask&(rgb.max(-1)<config['dark_rgb_max'])
    valid=mask&(depth>0);D=depth[valid]/np.maximum(a[valid],1e-8)
    small=cv2.resize(a.astype(np.float32),(config['shape_width'],config['shape_width']*9//16),interpolation=cv2.INTER_AREA)
    return dict(pixels=a.size,alpha_gt05_pixels=area,alpha_mass=float(a.sum()),
        dark_high_alpha_pixels=int(dark.sum()),dark_fraction_on_alpha=None if not area else float(dark.sum()/area),
        dark_alpha_mass=float(a[dark].sum()),missing_depth_on_alpha=int((mask&~valid).sum()),
        D_alpha_quantiles=None if not len(D) else np.quantile(D,[.1,.5,.9]).tolist(),
        shape_proxies=shape_proxies(small,threshold=config['alpha_threshold'],thin_radius=config['thin_radius'],
            island_pixels=config['island_pixels'],hole_pixels=config['hole_pixels']),
        independent_geometry_GT=False)

def angle_summary(rows,frames,config):
    bypair={(r['camera_index'],r['time_index']):r for r in rows};result={}
    for anchor in config['anchors']:
        values=[]
        for camera in config['all_camera_indices']:
            row=bypair[camera,anchor]
            distance=pose_distance(frames[anchor]['transform_matrix'],frames[camera]['transform_matrix'])
            values.append(dict(camera_index=camera,**distance,**row['dynamic']))
        bins=[]
        for lo,hi in zip(config['angle_bin_edges'][:-1],config['angle_bin_edges'][1:]):
            group=[r for r in values if lo<=r['view_direction_deg']<hi]
            valid=[r['dark_fraction_on_alpha'] for r in group if r['dark_fraction_on_alpha'] is not None]
            bins.append(dict(degrees=[lo,hi],views=len(group),valid_dark_ratios=len(valid),
                undefined_dark_ratios=len(group)-len(valid),
                dark_fraction_median=None if not valid else float(np.median(valid)),
                dark_fraction_min=None if not valid else float(np.min(valid)),
                dark_fraction_max=None if not valid else float(np.max(valid)),
                clipped_views=sum(r['shape_proxies']['touches_image_border'] for r in group)))
        result[str(anchor)]=dict(fixed_time=frames[anchor]['time'],matched=bypair[anchor,anchor]['dynamic'],
            spatial_views=values,angle_bins=bins,
            fixed_camera_time_controls=[dict(time_index=anchor+d,time=frames[anchor+d]['time'],
                time_difference=frames[anchor+d]['time']-frames[anchor]['time'],
                dynamic=bypair[anchor,anchor+d]['dynamic']) for d in config['offsets']])
    return result
