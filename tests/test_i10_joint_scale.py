import copy
import numpy as np
from variants.staged_reconstruction.i10_joint_scale import core

P=dict(smoothness=.05,huber_scale=.05,minimum_slope=1e-8,maximum_iterations=400,solver_tolerance=1e-10,ridge=1e-10)
C=dict(deduplicate_px=4,minimum_fit=6,minimum_check=4,minimum_split_fit=3,minimum_extent=.1,minimum_inverse_spread=.01,relative_median_max=.05,relative_P90_max=.1)
G=dict(minimum_overlap=2,maximum_separation_px=1.,deduplicate_px=4)


def fixture():
    y,x=np.mgrid[:64,:96];base=(1+x*.01+y*.003).astype(np.float32)
    uv=np.array([(x,y) for y in range(4,61,8) for x in range(4,93,8)],float)
    rows=[dict(id=str(j),group=str(j),split=['fit0','fit1','check'][j%3],uv=q.tolist(),
        depth=float(1/(.2*base[int(q[1]),int(q[0])]+.1))) for j,q in enumerate(uv)]
    raw=[(base*(1+.02*t)+.2*t).astype(np.float32) for t in range(12)]
    return [copy.deepcopy(rows) for _ in raw],raw,[core.normalize(q) for q in raw]


def test_joint_affine_drift_recovery():
    bank,raw,norms=fixture();q=core.joint_fit(bank,norms,raw,P,enabled=True)
    assert q['success']
    for t,c in enumerate(q['coefficients']):
        assert np.allclose(c,[.2/(1+.02*t),.1-.2*.2*t/(1+.02*t)],atol=1e-5)


def test_group_held_targets_never_enter_solver():
    bank,raw,norms=fixture();a=core.joint_fit(bank,norms,raw,P,enabled=True)
    for rows in bank:
        for q in rows:
            if q['split']=='check':q['depth']*=1.7
    b=core.joint_fit(bank,norms,raw,P,enabled=True)
    assert a==b


def test_input_immutability_and_replay():
    bank,raw,norms=fixture();original=copy.deepcopy(bank);saved=[q.copy() for q in raw]
    a=core.joint_fit(bank,norms,raw,P,enabled=True);b=core.joint_fit(bank,norms,raw,P,enabled=True)
    assert a==b and bank==original and all(np.array_equal(x,y) for x,y in zip(raw,saved))


def test_default_off_identity():
    a=object();assert core.joint_fit(None,None,None,None,original=a) is a


def test_no_fit_stays_unknown():
    result=core.joint_fit([[]]*12,[(0.,1.)]*12,[np.ones((8,8))]*12,P,enabled=True)
    assert not result['success'] and result['coefficients'] is None


def test_fitted_held_checks_and_positive_slopes():
    bank,raw,norms=fixture();solutions=[core.joint_fit(bank,norms,raw,P,half=h,enabled=True) for h in [None,0,1]]
    for t in range(12):
        cs=[q['coefficients'][t] for q in solutions]
        result=core.check_frame(raw[t],bank[t],cs[0],cs[1:],C)
        assert result['passed'] and cs[0][0]>0


def test_temporal_borrowing_cannot_override_missing_local_support():
    bank,raw,norms=fixture();bank[5]=[]
    q=core.joint_fit(bank,norms,raw,P,enabled=True)
    assert q['success']
    a=core.check_frame(raw[5],[],q['coefficients'][5],[q['coefficients'][5]]*2,C)
    assert not a['passed'] and a['reason']=='insufficient_anchors'


def test_held_contradiction_rejects_calibration():
    bank,raw,norms=fixture();cs=[core.joint_fit(bank,norms,raw,P,half=h,enabled=True)['coefficients'][0] for h in [None,0,1]]
    for q in bank[0]:
        if q['split']=='check':q['depth']*=1.5
    assert not core.check_frame(raw[0],bank[0],cs[0],cs[1:],C)['passed']


def test_framewise_diagnostic_exact_affine():
    bank,raw,_=fixture();assert core.framewise(raw[0],bank[0],C)['passed']


def track(name,start,x):
    return dict(id=name,start=start,uv=[[x,10.]]*12,
        stationarity=dict(label='stationary_compatible',stationary_hypothesis_xyz=[0.,0.,2.]))


def test_overlap_group_is_persistent_across_time():
    rows=[track('a',0,10),track('b',4,10.5),track('c',16,10.5)]
    groups=core.group_tracks(rows,G)
    assert groups['a']==groups['b'] and groups['c']!=groups['a']
    assert core.partition(groups['a'])==core.partition(groups['b'])


def test_partial_geometric_disagreement_blocks_merge():
    a=track('a',0,10);b=track('b',4,10.5);b['uv'][1]=[14.,10.]
    assert core.group_tracks([a,b],G)=={'a':'a','b':'b'}


def test_cell_dedup_does_not_inflate_counts_or_overlap_held():
    rows=[track(f'candidate{i}',0,10+i*.01) for i in range(12)]
    v=np.repeat(np.eye(4)[None],12,0)
    bank,grouping=core.anchor_bank(rows,v,np.eye(3),G)
    assert all(len(q)==1 for q in bank)
    assert sum(q['removed'] for q in grouping['deduplication'])==132


def test_coverage_degenerate_and_area():
    assert core.coverage([], (100,100))['hull_area_fraction']==0
    assert core.coverage([[0,0],[50,0],[50,50],[0,50]],(100,100))['hull_area_fraction']==.25


def test_feature_extraction_and_lk_deterministic():
    rng=np.random.default_rng(3);im=rng.integers(0,256,(64,96,3),dtype=np.uint8)
    p=dict(tiles=4,corners_per_tile=3,quality=.01,minimum_distance=4,block_size=5,window=21,pyramid_levels=3,iterations=30,epsilon=.01)
    a=core.features(im,p);assert np.array_equal(a,core.features(im,p))
    flow,status=core.flow([im]*4,a,p)
    assert status.all() and np.allclose(flow,a[None],atol=1e-4)


def test_flag_off_does_not_touch_training_model():
    import torch
    m=torch.nn.Linear(2,1);before={k:v.clone() for k,v in m.state_dict().items()}
    value=core.joint_fit(None,None,None,None,original=m)
    assert value is m and all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
