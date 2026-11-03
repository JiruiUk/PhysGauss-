import json
from pathlib import Path
import numpy as np
from variants.staged_reconstruction.depth_uncertainty.core import add_evidence,envelope,observations,pose_ids

P = json.loads((Path(__file__).resolve().parents[1]/'experiments/staged_reconstruction/S1_depth_uncertainty/protocol.json').read_text())


def fixture():
    views = np.repeat(np.eye(4)[None],7,axis=0)
    views[:,0,3] = [0,.2,.4,.6,-.2,-.4,-.6]
    k = np.array([[20.,0,31.5],[0,20,31.5],[0,0,1.]])
    return views,k,[np.full((64,64),4.,np.float64) for _ in range(7)]


def direct(z=4.,returned=None,available=None,reprojection=None,closer=None,ids=None,times=None):
    source = np.full((1,6),z) if returned is None else np.asarray(returned)[None].astype(float)
    avail = np.ones((1,6),bool) if available is None else np.asarray(available)[None]
    error = np.zeros((1,6)) if reprojection is None else np.asarray(reprojection)[None].astype(float)
    occlusion = np.zeros((1,6),bool) if closer is None else np.asarray(closer)[None]
    return envelope(np.array([z]),source,avail,error,occlusion,np.arange(6) if ids is None else np.asarray(ids),
        np.arange(1,7)*20 if times is None else np.asarray(times),0,1.98,6.83,P)


def test_flat_plane_geometry_and_exact_candidate_identity():
    views,k,depths = fixture(); z = np.array([4.]); uv = np.array([[32.,32.]])
    same,result = add_evidence(z,uv,views,k,0,list(range(1,7)),depths,list(range(0,140,20)),1.98,6.83,P,True)
    assert same is z and result['tight'].all() and result['available_sources'][0] == 6
    assert np.allclose(result['source_returned_depth'],4.) and result['lower'][0] < 4 < result['upper'][0]


def test_default_off_preserves_original_object():
    z = np.array([4.]); same,result = add_evidence(z)
    assert same is z and result is None


def test_empty_sources_are_unknown_not_zero_depth():
    r = direct(available=[False]*6)
    assert not r['tight'].any() and r['confidence_score'][0] == 0
    assert 0 < r['lower'][0] < 4 < r['upper'][0]


def test_envelope_contains_all_valid_disagreement():
    r = direct(returned=[3.,4.,4.1,5.,4.,4.])
    assert r['lower'][0] < 3 and r['upper'][0] > 5 and not r['tight'].any()


def test_closer_surface_is_ambiguous_not_deleted():
    r = direct(closer=[True,False,False,False,False,False])
    assert r['closer_surface_unknown_count'][0] == 1 and not r['tight'].any()


def test_occluder_geometry_widens_interval():
    views,k,depths = fixture(); depths[1][:] = 2.
    z,r = add_evidence(np.array([4.]),np.array([[32.,32.]]),views,k,0,list(range(1,7)),depths,list(range(0,140,20)),1.98,6.83,P,True)
    assert z[0] == 4 and r['closer_surface_unknown_count'][0] == 1
    assert r['lower'][0] < 2 and not r['tight'].any()


def test_duplicate_camera_poses_do_not_inflate_evidence():
    r = direct(ids=[0,0,0,0,0,0]); assert r['distinct_source_poses'][0] == 1 and not r['tight'].any()
    views,_,_ = fixture(); views[2] = views[1]
    assert pose_ids(views)[1] == pose_ids(views)[2]


def test_temporal_span_is_required():
    r = direct(times=[1,2,3,4,5,6]); assert r['source_time_span'][0] == 6 and not r['tight'].any()


def test_reprojection_bound_is_not_hidden_by_median():
    r = direct(reprojection=[0,0,0,0,0,2.])
    assert r['maximum_reprojection_px'][0] == 2 and not r['tight'].any()


def test_nonfinite_source_depth_is_ignored():
    r = direct(returned=[float('nan'),4,4,4,4,4])
    assert r['available_sources'][0] == 5 and np.isfinite(r['lower']).all()


def test_out_of_view_or_missing_sources_give_no_negative_vote():
    views,k,depths = fixture(); views[1,0,3] = 100; depths[2][:] = 0
    returned,available,_,closer = observations(np.array([[32.,32.]]),np.array([4.]),views,k,0,list(range(1,7)),depths)
    assert not available[0,:2].any() and not closer[0,:2].any() and (returned[0,:2] == 0).all()


def test_grid_spacing_floor_and_wider_depth_abstention():
    r = direct(); assert 0 < r['width'][0] <= P['maximum_tight_interval_width']
    r = direct(z=6.); assert r['width'][0] > P['maximum_tight_interval_width'] and not r['tight'].any()


def test_camera_identity_normalizes_signed_zero():
    views = np.repeat(np.eye(4)[None],2,axis=0); views[1,0,3] = -0.
    assert pose_ids(views)[0] == pose_ids(views)[1]
