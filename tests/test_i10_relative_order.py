import numpy as np
from variants.staged_reconstruction.i10_relative_order.core import make_pairs,patch

P=dict(maximum_raw_patch_range=.1,minimum_pixel_separation=8,raw_normalized_gap_min=.1,
       minimum_order_fit_observations=4,minimum_order_check_observations=4,maximum_pairs_per_frame=4,pair_confidence=.5)


def fixture():
    y,x=np.mgrid[:64,:96];raw=(1+x*.02+y*.001).astype(np.float32)
    frames=[raw*(1+t*.1)+t for t in range(12)]
    row=dict(id='native_eligible',row=5,start=0,uv=[[20,30]]*12,passed=False)
    anchors=[[dict(id='anchor',group='persistent',split='fit0',uv=[80,30])]]*12
    a=dict(candidate=dict(motion=[row]),anchor_bank=anchors,
        normalizations=[(float(np.median(r)),float(np.quantile(r,.9)-np.quantile(r,.1))) for r in frames])
    return a,frames


def test_metric_depth_not_needed_positive_affine_invariance():
    a,raw=fixture();q=make_pairs(a,raw,P)
    assert len(q['pairs'])==6 and all(p['sign']==-1 for p in q['pairs'])
    assert all(not p['frame']%2 for p in q['pairs'])
    assert not a['candidate']['motion'][0]['passed'] # failed metric coverage cannot masquerade as metric admission


def test_held_can_reject_but_never_choose_sign():
    a,raw=fixture();raw=[np.fliplr(r).copy() if t%2 else r for t,r in enumerate(raw)]
    q=make_pairs(a,raw,P);assert q['pairs']==[]
    assert q['hypotheses'][0]['fit_chosen_sign']==-1 and q['hypotheses'][0]['reason']=='held_order_conflict'


def test_fit_conflict_preserved():
    a,raw=fixture();raw[2]=np.fliplr(raw[2]).copy();q=make_pairs(a,raw,P)
    assert q['pairs']==[] and q['hypotheses'][0]['reason']=='fit_order_conflict'


def test_insufficient_repeated_support_unknown():
    a,raw=fixture();a['anchor_bank']=[[] if i<5 else x for i,x in enumerate(a['anchor_bank'])]
    assert make_pairs(a,raw,P)['pairs']==[]


def test_check_anchor_groups_excluded_from_training():
    a,raw=fixture()
    for rows in a['anchor_bank']:
        for q in rows:q['split']='check'
    assert make_pairs(a,raw,P)['pairs']==[] and len(make_pairs(a,raw,P,True)['pairs'])==6


def test_patch_edge_missing_and_strong_discontinuity():
    image=np.ones((64,96),np.float32);image[:,48:]=3
    assert patch(image,[0,0],2,P) is None and patch(image,[48,30],2,P) is None
    assert patch(image,[20,30],2,P)==(1.,1.)


def test_replay_and_no_binding_means_no_target():
    a,raw=fixture();assert make_pairs(a,raw,P)==make_pairs(a,raw,P)
    del a['candidate']['motion'][0]['row'];assert make_pairs(a,raw,P)['pairs']==[]
