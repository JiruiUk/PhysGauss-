import torch
from variants.integrity_feedback.model import foreground_tail_loss
from variants.integrity_feedback.policy import propose,retry,review_final


def test_tail_updates_worst_foreground_and_ignores_background():
    image=torch.full((3,4,4),.5,requires_grad=True);target=torch.full_like(image,.5)
    mask=torch.zeros(1,4,4);mask[:,:,1:3]=1
    target[:,1,1]=1.;target[:,0,0]=1.
    loss=foreground_tail_loss(image,target,mask);loss.backward()
    assert image.grad[:,1,1].sum()<0 and image.grad[:,0,0].sum()==0
    assert torch.isfinite(image.grad).all()


def test_empty_foreground_has_zero_finite_gradient():
    image=torch.rand(3,4,4,requires_grad=True)
    loss=foreground_tail_loss(image,torch.zeros_like(image),torch.zeros(1,4,4));loss.backward()
    assert loss==0 and image.grad.abs().sum()==0


def evidence():
    metrics={'PSNR':39.,'SSIM':.99,'LPIPS':.02,'MS-SSIM':.995}
    geometry={'integrity_score':.02,'p95_score':.03,'background_false_coverage':.01,
        'opaque_appearance_error':.01,'core_low_50':.0001,'core_low_90':.002,'fragment_loss':.003}
    config={'integrity_coverage_weight':.002,'integrity_strain_weight':0.,'coverage_density':False}
    row={'model':'screen','metrics':metrics,'integrity':geometry}
    return metrics,geometry,config,row


def test_policy_routes_opaque_error_to_rgb_loss_not_alpha_increase():
    m,g,c,r=evidence();p=propose([r],{'screen':c},m,g)
    assert p['action']=='train' and p['evidence_split']=='validation'
    assert p['knobs']['integrity_tail_weight']>0
    assert p['knobs']['integrity_coverage_weight']<=c['integrity_coverage_weight']
    assert c['integrity_coverage_weight']==.002


def test_unsafe_screens_cannot_enable_aggressive_density():
    m,g,c,r=evidence();r=dict(r,metrics=dict(m,PSNR=37.));c=dict(c,coverage_density=True)
    p=propose([r],{'screen':c},m,g)
    assert not p['knobs']['coverage_density'] and p['knobs']['integrity_coverage_weight']<=.0005


def test_retry_responds_to_measured_quality_drop_and_is_bounded():
    m,g,c,r=evidence();p=propose([r],{'screen':c},m,g)
    result=dict(r,metrics=dict(m,PSNR=37.))
    q=retry(p,result,m,g)
    assert q['knobs']['integrity_tail_weight']<p['knobs']['integrity_tail_weight']
    assert not q['knobs']['coverage_density'] and q['maximum_new_long_runs']==1


def test_completed_validation_is_rechecked_without_using_test():
    m,g,c,r=evidence();c=dict(c,coverage_density=True)
    p=propose([r],{'screen':c},m,g)
    q=review_final(p,{'metrics':dict(m,PSNR=37.),'integrity':dict(g,background_false_coverage=.1)},m,g)
    assert not q['knobs']['coverage_density']
    assert q['knobs']['integrity_coverage_weight']<p['knobs']['integrity_coverage_weight']
    assert q['evidence_split']=='validation'
