import pytest
import torch
from variants.temporal_repair.model import bracket,interpolate_query,nlerp,repaired_state
from variants.kinematic_tuning.model import center_trajectory


def state(x,t):
    q=x.new_tensor([1.,0,0,0]).expand(len(x),-1)
    return {'d_xyz':torch.cat((q,torch.ones_like(x)*t),-1),'d_rotation':q*0,'d_scaling':x*0}


def test_flags_off_queries_once_and_returns_identical_object():
    x=torch.randn(8,3);calls=[];original=state(x,.2)
    def query(x,t):calls.append(t);return original
    actual=interpolate_query(query,x,.2,[0.,1.],x,x,strength=0.)
    assert actual is original and calls==[.2]


def test_training_knots_and_outside_support_preserve_original():
    x=torch.randn(8,3);s=state(x,0);calls=[]
    def query(x,t):calls.append(t);return s
    for t in [0.,.2,.5,1.]:
        calls.clear();assert interpolate_query(query,x,t,[.2,.5,.8],x,x,1.) is s
        assert calls==[t]
    with pytest.raises(ValueError):bracket([0.,0.],.1)
    with pytest.raises(ValueError):bracket([0.,1.],float('nan'))


def test_affine_center_motion_and_covariance_are_preserved():
    x=torch.randn(8,3);q=x.new_tensor([1.,0,0,0]).expand(8,-1);scale=torch.ones_like(x)
    for full in [False,True]:
        actual=interpolate_query(state,x,.3,[0.,1.],scale,q,1.,full)
        assert torch.allclose(center_trajectory(x,actual['d_xyz']),x+.3,atol=2e-7)
        assert torch.equal(actual['d_scaling'],x*0)


def test_center_repair_is_not_raw_quaternion_translation_lerp():
    x=torch.tensor([[1.,0,0]]);a=state(x,0);b=state(x,0);b['d_xyz'][0,:4]=torch.tensor([0.,0,0,1.])
    result=repaired_state(x,torch.ones_like(x),a['d_xyz'][:,:4],a,a,b,.5,1.)
    assert torch.allclose(center_trajectory(x,result['d_xyz']),torch.zeros_like(x))


def test_quaternion_sign_ambiguity_does_not_collapse_rotation():
    q=torch.tensor([[.7,0,.7,0]])
    assert torch.allclose(nlerp(q,-q,.5),torch.nn.functional.normalize(q,dim=-1))


def test_raw_covariance_interpolation_preserves_nonunit_stationary_rotations():
    x=torch.randn(8,3);s=state(x,0);q=s['d_xyz'][:,:4].clone();s['d_rotation']=q*.6
    actual=repaired_state(x,torch.ones_like(x),q,s,s,s,.5,1.,True)
    assert torch.allclose(actual['d_rotation'],s['d_rotation'],atol=1e-7)


def test_disabled_runtime_returns_parent_result_without_scene_access(tmp_path,monkeypatch):
    import json
    from variants.temporal_repair.runtime import Tuning,Parent
    p=tmp_path/'config.json';p.write_text(json.dumps({'enabled':False,'integrity_hard_frame_probability':.25}))
    expected=object();monkeypatch.setattr(Parent,'coverage_loss',lambda *args:expected)
    tuning=Tuning(p,tmp_path)
    assert tuning.coverage_loss(100,object(),object(),object(),object()) is expected
    assert tuning.select_training_camera(100,expected,[]) is expected


def test_unit_covariance_quaternion_handles_degenerate_input_and_has_finite_gradient():
    from variants.temporal_repair.covariance import unit_quaternion
    raw=torch.tensor([[1.6,.2,.3,.4],[0.,0,0,0]],requires_grad=True)
    assert unit_quaternion(raw,False) is raw
    result=unit_quaternion(raw,True)
    assert torch.allclose(result.norm(dim=-1),torch.ones(2)) and torch.equal(result[1],torch.tensor([1.,0,0,0]))
    result[:,1].sum().backward();assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0


def test_sampler_rejects_heldout_observations_and_off_consumes_no_rng():
    import random
    from types import SimpleNamespace
    from variants.temporal_repair.sampling import FrameSampler
    a=FrameSampler(['0002','0003'],.25)
    with pytest.raises(ValueError):a.observe('0084',.1)
    with pytest.raises(ValueError):a.observe('0002',float('nan'))
    current=SimpleNamespace(image_name='0002');before=random.getstate()
    assert a.select(current,[],enabled=False) is current and random.getstate()==before


def test_sampler_roundtrip_preserves_rng_policy_and_every_candidate_is_training():
    import json,random
    from types import SimpleNamespace
    from variants.temporal_repair.sampling import FrameSampler
    views=[SimpleNamespace(image_name=str(i)) for i in range(12)]
    a=FrameSampler([v.image_name for v in views],.25)
    for v in views:
        a.observe(v.image_name,float(v.image_name));a.observe(v.image_name,float(v.image_name))
    saved=json.loads(json.dumps(a.state()));rng=random.getstate()
    expected=[a.select(views[0],views).image_name for _ in range(200)]
    b=FrameSampler(a.names,.25);b.restore(saved);random.setstate(rng)
    assert [b.select(views[0],views).image_name for _ in range(200)]==expected
    assert b.replacements>0 and set(expected)<={'0','9','10','11'}
