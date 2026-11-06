from copy import deepcopy
import numpy as np
import pytest
from variants.staged_reconstruction.view_time_audit.core import pairs,pose_distance,describe,angle_summary

P=dict(alpha_threshold=.5,dark_rgb_max=.05,shape_width=400,thin_radius=2,island_pixels=20,hole_pixels=4)

def test_disabled_never_inspects_poison_inputs():
    assert describe(object(),object(),object(),object()) is None

def test_empty_support_is_undefined_not_perfect():
    a=np.zeros((90,160));q=describe(a,np.zeros(a.shape+(3,)),a,P,enabled=True)
    assert q['alpha_gt05_pixels']==0 and q['dark_fraction_on_alpha'] is None
    assert q['D_alpha_quantiles'] is None and not q['shape_proxies']['defined']

def test_denominators_darkness_depth_and_clipping_are_separate():
    a=np.zeros((90,160));a[20:60,30:90]=.8;rgb=np.repeat(a[...,None],3,2)*.5
    rgb[20:40,30:90]=0;depth=a*4;before=deepcopy((a,rgb,depth))
    q=describe(a,rgb,depth,P,enabled=True)
    assert q['alpha_gt05_pixels']==2400 and q['dark_high_alpha_pixels']==1200
    assert q['dark_fraction_on_alpha']==.5 and q['D_alpha_quantiles']==[4.,4.,4.]
    assert not q['shape_proxies']['touches_image_border']
    assert all(np.array_equal(x,y) for x,y in zip(before,(a,rgb,depth)))
    a[:,0:5]=.9
    assert describe(a,rgb,depth,P,enabled=True)['shape_proxies']['touches_image_border']

def test_pose_distance_translation_and_rotation_do_not_mix():
    a=np.eye(4);b=a.copy();b[:3,3]=[3,4,0]
    q=pose_distance(a,b);assert q['camera_translation']==5 and q['rotation_deg']==0 and q['view_direction_deg']==0
    b=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1.]])
    q=pose_distance(a,b);assert q['rotation_deg']==pytest.approx(90) and q['view_direction_deg']==pytest.approx(90)
    assert pose_distance(a,b)==pose_distance(b,a)

def test_pair_union_keeps_crossing_and_exact_diagonal_once():
    p=dict(anchors=[2,4],offsets=[-1,0,1],all_camera_indices=list(range(7)))
    actual=pairs(p);expected={(c,t) for t in [2,4] for c in range(7)}
    expected|={(c,t) for k in [2,4] for c in [k-1,k,k+1] for t in [k-1,k,k+1]}
    assert set(actual)==expected and len(actual)==len(set(actual))
    assert all((i,i) in actual for i in [1,2,3,4,5])
    assert p['offsets']==[-1,0,1]

def test_nonfinite_data_not_silently_scored():
    a=np.zeros((90,160));a[20,20]=np.nan
    with pytest.raises(AssertionError):describe(a,np.zeros(a.shape+(3,)),np.zeros_like(a),P,enabled=True)

def test_angle_bins_account_for_all_empty_ratios():
    p=dict(anchors=[0],offsets=[0],all_camera_indices=[0,1],angle_bin_edges=[0,5,180.000001])
    frames=[dict(time=0.,transform_matrix=np.eye(4).tolist()),dict(time=1.,transform_matrix=np.eye(4).tolist())]
    dynamic=describe(np.zeros((90,160)),np.zeros((90,160,3)),np.zeros((90,160)),P,enabled=True)
    rows=[dict(camera_index=i,time_index=0,dynamic=dynamic) for i in [0,1]]
    q=angle_summary(rows,frames,p)['0'];assert q['angle_bins'][0]['views']==2
    assert q['angle_bins'][0]['valid_dark_ratios']==0 and q['angle_bins'][0]['dark_fraction_median'] is None

def test_frozen_protocol_pair_count_and_crossing():
    import json
    from pathlib import Path
    p=json.loads(Path('experiments/staged_reconstruction/RubberDuck_view_time_audit/protocol.json').read_text())
    actual=pairs(p)
    assert len(actual)==482
    assert len(set(actual))==len(actual)
    for anchor in p['anchors']:
        assert sum(t==anchor for c,t in actual)==120

def test_state_digest_handles_full_nested_tensor_numpy_state():
    import torch
    from variants.staged_reconstruction.view_time_audit.worker import state_digest
    a=dict(model=torch.tensor([1.,2.]),rng=(np.array([1,2]),3),flags=[True,False])
    assert state_digest(a)==state_digest(deepcopy(a))
    b=deepcopy(a);b['model'][0]=4
    assert state_digest(a)!=state_digest(b)

def test_default_off_cli_does_not_load_cuda_model():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-m','variants.staged_reconstruction.view_time_audit.worker'],capture_output=True,text=True)
    assert result.returncode==0 and result.stdout.strip()=='V0_DISABLED'
