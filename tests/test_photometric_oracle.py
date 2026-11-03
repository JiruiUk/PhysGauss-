import subprocess
import sys
import numpy as np
import torch
from variants.staged_reconstruction.photometric_oracle.core import (scene_hit,single_face,world_points,
    nearest_grid,measure,true_visibility,controls)
from variants.staged_reconstruction.photometric_oracle.stats import groups,summarize,merge,compact
from variants.staged_reconstruction.synthetic_audit.generate import trace,camera_view


def test_cli_default_off_no_run_directory_access():
    r=subprocess.run([sys.executable,'-m','variants.staged_reconstruction.photometric_oracle.audit'],capture_output=True,text=True)
    assert r.returncode != 0 and 'opt-in' in r.stderr


def test_measure_default_off_identity_without_rgb_or_GT():
    original=torch.tensor([[4.]])
    assert measure(None,original) is original


def test_nearest_grid_formula_clamping_and_continuous_reference():
    z=torch.tensor([1.,2.,3.,4.,6.,7.]); actual=nearest_grid(z,2,6)
    grid=1/torch.linspace(.5,1/6,192)
    expected=grid[((1/z-.5)/((1/6-.5)/191)).round().long().clamp(0,191)]
    assert torch.equal(actual,expected) and actual[0]==2 and actual[-1]==6


def test_single_face_rejects_border_missing_and_normal_boundary():
    d=np.ones((9,9)); n=np.zeros((9,9,3)); n[...,2]=1; v=np.ones((9,9),bool)
    out=single_face(d,n,v); assert out.sum()==25 and not out[:2].any()
    v[4,4]=False; assert not single_face(d,n,v)[2:7,2:7].any()
    v[:]=True; n[4,4]=[1,0,0]; assert not single_face(d,n,v)[2:7,2:7].any()


def test_analytic_hit_matches_generator_observed_and_static():
    v=camera_view(.7); k=np.array([[35.,0,31.5],[0,35.,31.5],[0,0,1.]])
    y,x=np.mgrid[:64,:64]; uv=np.c_[x.ravel(),y.ravel()]
    inv=np.linalg.inv(v); rays=(np.c_[uv,np.ones(len(uv))]@np.linalg.inv(k).T)@inv[:3,:3].T
    for static in [False,True]:
        d=scene_hit(inv[:3,3],rays,.4,static)
        gt=trace(v,k,64,64,.4,static_only=static)['depth'].ravel()
        assert np.array_equal(np.isfinite(d),gt>0)
        assert np.max(np.abs(d[gt>0]-gt[gt>0]))<1e-6


def test_true_visibility_t_one_and_occlusion():
    # Camera at (0,0,4), looking -z: pillar front point is visible, back is occluded.
    v=np.eye(4); v[0,0]=-1; v[2,2]=-1; v[2,3]=4
    views=np.stack([v]); vt=torch.tensor(views,dtype=torch.float32)
    k=torch.tensor([[35.,0,31.5],[0,35.,31.5],[0,0,1.]])
    points=np.array([[0.,0.,.2],[0.,0.,-.2],[4.,0.,.2]])
    visible=true_visibility(points,views,np.array([0.]),[0],vt,k,64,64)
    assert visible.tolist()==[[True,False,False]]


def test_backprojection_world_reprojects_expected_depth():
    uv=np.array([[32.,30.],[29.,35.]]); k=np.array([[35.,0,31.5],[0,35.,31.5],[0,0,1.]])
    view=camera_view(.2); z=np.array([4.,4.2]); points=world_points(uv,z,view,k)
    local=points@view[:3,:3].T+view[:3,3]; q=local@k.T
    assert np.allclose(local[:,2],z) and np.allclose(q[:,:2]/q[:,2:],uv)


def test_groups_source_subset_and_cross_partition():
    vis=np.ones((6,4),bool); vis[0,1]=False; vis[4,2]=False; vis[:,3]=False
    gs=groups(np.array([True,False,True,False]),vis,np.array([.01,.03,.06,.08]),np.array([False,True,False,False]))
    assert gs['all6_visible_subset'].sum()==1 and gs['fit3_bothheld_visible_inclusive'].sum()==2
    cross=[v for key,v in gs.items() if '/' in key]
    assert np.array_equal(np.stack(cross).sum(0),np.ones(4))


def test_stats_merging_has_correct_denominators():
    c=np.full((3,6,4),.3,dtype=np.float32); c[1]=.2; c[2]=1
    f=c[:,:3].mean(1); a=np.array([[True]*4,[False]*4]); mask=np.ones(4,bool)
    full=summarize(c,f,a,mask)
    merged=merge([{'all':summarize(c,f,a,np.array([True,True,False,False]))},
                  {'all':summarize(c,f,a,np.array([False,False,True,True]))}])['all']
    assert full==merged and full['unit_source_costs']==[0,0,24]
    assert compact({'all':full})['all']['metrics']['GT_fit']['mean']<.21


def test_analytic_three_controls_preserve_fractional_fixture_bias():
    cs=controls(); assert set(cs)=={'A_fractional','B_integer','C_slanted'}
    # These are mechanism checks, not fitting a fixture to suppress the original bias.
    a,b=cs['A_fractional'],cs['B_integer']
    assert len(a['fit_curve'])==2 and len(a['fit_curve'][0])==192
    assert max(abs(v-4) for v in b['minimum_depth'])<.03
    assert max(abs(v-4) for v in a['minimum_depth'])>.03
    assert cs['C_slanted']['true_depth'][0]!=cs['C_slanted']['true_depth'][1]


def test_measure_uses_original_four_best_three_and_acceptance(monkeypatch):
    import variants.staged_reconstruction.photometric_oracle.core as core
    def cost(points,reference,source,*args):
        j=int(source[0,0,0]); z=points[...,2]
        return torch.where(z<4.5,torch.full_like(z,.05 if j!=2 else .9),torch.full_like(z,.2))
    monkeypatch.setattr(core,'source_cost',cost)
    feat=torch.ones(7,28,8,8)
    for j in range(7): feat[j,0]=j
    c,f,a=measure(torch.tensor([[3.,3.]]),torch.tensor([[4.8],[4.],[4.1]]),feat,
                  torch.ones(7,8,8,dtype=torch.bool),torch.eye(4).repeat(7,1,1),torch.eye(3),0,list(range(1,7)),True)
    assert torch.allclose(f[1],torch.tensor([.05])) and a.all()


def test_distribution_empty_is_missing_not_zero_error():
    c=np.zeros((3,6,1)); f=np.zeros((3,1)); a=np.zeros((2,1),bool)
    g=compact({'none':summarize(c,f,a,np.array([False]))})['none']
    assert g['n']==0 and g['metrics']['GT_fit']['mean'] is None


def test_original_invalid_projection_is_unit_cost():
    from variants.staged_reconstruction.adaptive_static.geometry import patch_features,source_cost
    torch.manual_seed(0); feat,texture=patch_features(torch.rand(1,3,16,16),5)
    ref=feat[0,:,8,8,None]
    c=source_cost(torch.tensor([[[100.,100.,1.]]]),ref,feat[0],texture[0],torch.eye(4),torch.eye(3))
    assert c.item()==1


def test_independent_process_aggregate_save_resume(tmp_path):
    # Only aggregate state, no privileged pixel arrays, survives the process boundary.
    from variants.staged_reconstruction.photometric_oracle.audit import save_or_verify
    c=np.full((3,6,4),.3); f=c[:,:3].mean(1); a=np.zeros((2,4),bool)
    one={'all':summarize(c,f,a,np.array([True,True,False,False]))}
    two={'all':summarize(c,f,a,np.array([False,False,True,True]))}
    import json
    p1,p2,out=[tmp_path/n for n in ['one.json','two.json','out.json']]
    save_or_verify(p1,one); save_or_verify(p2,two)
    code=('import json,sys; from pathlib import Path; '
          'from variants.staged_reconstruction.photometric_oracle.stats import merge; '
          'from variants.staged_reconstruction.photometric_oracle.audit import save_or_verify; '
          'p=list(map(Path,sys.argv[1:])); '
          'save_or_verify(p[2],merge([json.loads(p[0].read_text()),json.loads(p[1].read_text())]))')
    for _ in range(2):
        r=subprocess.run([sys.executable,'-c',code,str(p1),str(p2),str(out)],capture_output=True,text=True)
        assert r.returncode==0,r.stderr
    assert json.loads(out.read_text())==merge([one,two])
