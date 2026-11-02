import numpy as np
from output.short_clip_diagnostic.camera_math import w2c,intrinsic,project,fundamental,sampson,motion_split


def scene():
    rng=np.random.default_rng(381);points=rng.uniform([-1,-1,3],[1,1,6],(100,3));a=np.eye(4);b=np.eye(4)
    t=.12;b[:3,:3]=[[np.cos(t),0,np.sin(t)],[0,1,0],[-np.sin(t),0,np.cos(t)]];b[:3,3]=[.2,.03,-.04]
    return points,a,b,intrinsic(.85,'metadata')


def test_true_static_correspondences_satisfy_epipolar_geometry():
    p,a,b,k=scene();x,_=project(p,a,k);y,_=project(p,b,k)
    assert np.max(sampson(x,y,fundamental(a,b,k,k)))<1e-9


def test_wrong_intrinsics_can_be_detected_without_depth():
    p,a,b,k=scene();x,_=project(p,a,k);y,_=project(p,b,k);wrong=intrinsic(.85,'legacy')
    assert np.median(sampson(x,y,fundamental(a,b,wrong,wrong)))>.01


def test_sampson_invariant_to_matrix_scale():
    p,a,b,k=scene();x,_=project(p,a,k);y,_=project(p,b,k);y+=.5;f=fundamental(a,b,k,k)
    assert np.allclose(sampson(x,y,f),sampson(x,y,1e-12*f),atol=1e-12)


def test_zero_baseline_is_not_false_perfect_score():
    p,a,b,k=scene();assert fundamental(a,a,k,k) is None
    assert np.isnan(sampson(np.zeros((2,2)),np.zeros((2,2)),None)).all()


def test_legacy_and_metadata_translation_are_explicit_different_hypotheses():
    c=np.eye(4);c[:3,3]=[1,2,3]
    assert np.allclose(w2c(c,'legacy')[:3,3],[1,2,3])
    assert np.allclose(w2c(c,'metadata_cv')[:3,3],[-1,2,3])


def test_camera_and_object_split_sums_to_full_projection_motion():
    p,a,b,k=scene();q=p.copy();q[:,0]+=.02
    full,camera,obj=motion_split(p,q,a,b,k)
    assert np.allclose(full,camera+obj,atol=1e-12)
    static,cam,obj=motion_split(p,p,a,b,k)
    assert np.allclose(obj,0) and np.allclose(static,cam)
