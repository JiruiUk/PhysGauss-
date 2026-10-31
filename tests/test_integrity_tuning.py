import numpy as np
import torch
from variants.integrity_tuning.model import coverage_objective,strain_objective,persistent_edges,image_metrics


def test_coverage_fills_core_and_penalizes_background():
    target=torch.zeros(1,15,15);target[:,2:13,2:13]=1
    alpha=(target*.4+.1).requires_grad_()
    coverage_objective(alpha,target).backward()
    assert alpha.grad[0,7,7]<0 and alpha.grad[0,0,0]>0


def test_empty_foreground_has_finite_loss_and_gradient():
    a=torch.zeros(1,10,10,requires_grad=True)
    loss=coverage_objective(a,torch.zeros_like(a));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all()


def geometry():
    x=torch.tensor([[0.,0,0],[1.,0,0],[0,1.,0]])
    return x,torch.tensor([0]),torch.tensor([[1,2]]),torch.ones(1,2,dtype=torch.bool)


def test_rigid_motion_and_allowed_stretch_have_zero_strain():
    x,a,n,v=geometry();r=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]])
    assert strain_objective(x,x@r+3,a,n,v)<1e-10
    assert strain_objective(x,x*1.04,a,n,v)==0


def test_excessive_separation_has_finite_restoring_gradient():
    x,a,n,v=geometry();y=(x*1.5).requires_grad_()
    loss=strain_objective(x,y,a,n,v);loss.backward()
    assert loss>0 and y.grad[1,0]>0 and torch.isfinite(y.grad).all()


def test_missing_edges_give_zero_and_persistent_gate_rejects_transient_contact():
    x,a,n,v=geometry();y=x.clone();y[1,0]=10
    gate=persistent_edges([x,y],a,n,v,2.)
    assert not gate[0,0] and gate[0,1]
    assert strain_objective(x,y,a,n,torch.zeros_like(v))==0


def test_opaque_bad_color_is_not_mislabeled_as_alpha_hole():
    target=np.ones((20,20));rgb=np.zeros((20,20,3));gt=np.ones_like(rgb)
    m=image_metrics(target,target,rgb,gt)
    assert m['core_low_50']==0 and m['opaque_appearance_error']==1


def test_fragmentation_and_background_spill_are_distinguished():
    mask=np.zeros((25,25));mask[3:22,3:22]=1
    rgb=np.ones((25,25,3));alpha=mask.copy();alpha[11:14,:]=0
    m=image_metrics(alpha,mask,rgb,rgb)
    assert m['extra_fragments']>=1 and m['core_low_50']>0
    assert m['background_false_coverage']==0
    assert image_metrics(np.ones_like(mask),mask,rgb,rgb)['background_false_coverage']==1


def test_selection_state_checkpoint_round_trip(tmp_path):
    import json
    from variants.integrity_tuning.runtime import Tuning
    from variants.metric_tuning.runtime import sha256
    config=tmp_path/'config.json';config.write_text(json.dumps({'enabled':True,'integrity_enabled':True}))
    checkpoint=tmp_path/'resume.pt';checkpoint.write_bytes(b'full-state-checksum-fixture')
    a=Tuning(config,tmp_path);a.integrity_best={'iteration':42,'score':.1,'snapshot':'integrity_best'}
    a.save(checkpoint,{'sha256':sha256(checkpoint),'iteration':42})
    b=Tuning(config,tmp_path);b.restore(checkpoint,42)
    assert b.integrity_best==a.integrity_best


def test_promotion_rejects_background_inflation_and_image_regression(tmp_path,monkeypatch):
    import json
    from variants.integrity_tuning import run
    reference={'PSNR':39.,'SSIM':.99,'LPIPS':.02,'MS-SSIM':.995}
    monkeypatch.setattr(run,'REPORTS',tmp_path)
    monkeypatch.setattr(run.base,'history',lambda _: [{'iteration':20000,'metrics':reference}])
    (tmp_path/'diagnosis.json').write_text(json.dumps({'mean':{'background_false_coverage':.01}}))
    def candidate(name,psnr,spill,score):
        return {'model':name,'metrics':dict(reference,PSNR=psnr),
            'integrity':{'integrity_score':score,'p95_score':score,'background_false_coverage':spill}}
    safe=candidate('safe',38.8,.012,.1)
    opaque_blob=candidate('inflated',39.,.5,0.)
    image_failure=candidate('bad_RGB',38.,.012,0.)
    assert run.select([opaque_blob,image_failure,safe])==['safe']


def test_branch_receipt_passes_real_checkpoint_relocation_contract(tmp_path):
    from variants.integrity_tuning import run
    from variants.kinematic_resplit.checkpoint import validate_relocation
    source,destination=tmp_path/'source',tmp_path/'destination'
    for directory in (source,destination):
        directory.mkdir();(directory/'model_config.py').write_text('same_original_recipe=True\n')
    for folder in ('point_cloud','deform','checkpoints'):(source/folder).mkdir()
    checkpoint=source/'checkpoints/resume.pt';checkpoint.write_bytes(b'identical-full-state-fixture')
    digest=run.engine.sha256(checkpoint)
    run.engine.write_json(source/'checkpoints/resume.json',{'iteration':20,'sha256':digest})
    a={'integrity_enabled':False};b={'integrity_enabled':True}
    run.engine.write_json(source/'metric_config.json',a);run.engine.write_json(destination/'metric_config.json',b)
    run.engine.write_json(source/'checkpoints/resume.metrics.json',{'config':a})
    run.engine.write_json(source/'metric_best.json',{});(source/'metric_history.jsonl').write_text('')
    run.branch(source,destination,20)
    receipt=run.engine.read_json(destination/'branch.json')
    saved={'save_image_dir':str(source/'diagnostics'),'unchanged_option':1}
    current=dict(saved,save_image_dir=str(destination/'diagnostics'))
    assert validate_relocation(saved,current,receipt,digest)
    # Repair an already-created receipt from the failed preflight safely.
    receipt.pop('destination');run.engine.write_json(destination/'branch.json',receipt)
    run.branch(source,destination,20)
    assert validate_relocation(saved,current,run.engine.read_json(destination/'branch.json'),digest)
    assert run.engine.sha256(destination/'checkpoints/resume.pt')==digest
