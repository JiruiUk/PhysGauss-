"""Real native rendering: gradient isolation, trainable geometry and dynamic CMR."""
import os
import sys
import numpy as np
import torch
from PIL import Image
from .run import ROOT, REPORT, RUNS, DATA, source, configuration, engine
from .prepare import prepare


def geometry(g, w):
    from variants.kinematic_tuning.model import center_trajectory
    from variants.flow_comparison.flow import rotation_raw
    from .render import apply_warp
    a,b,c = apply_warp(g, w['d_xyz'], w['d_rotation'], w['d_scaling'])
    rot = rotation_raw(g.get_rotation+b); scale = g.get_scaling+c
    return center_trajectory(g.get_xyz,a), (rot*scale[:,None,:].square())@rot.transpose(-1,-2)


def main():
    torch.set_num_threads(4); repo = prepare(); sys.path.insert(0,str(repo))
    from variants.association_training.evaluate import load, render_call
    from variants.shape_coherence.observations import camera
    from variants.flow_comparison.temporal import warp_at
    from .render import pool_for, physics_candidates
    folder = RUNS/'_probe'; folder.mkdir(parents=True,exist_ok=True)
    config = configuration(True); engine.write_json(folder/'metric_config.json',config)
    os.environ['PIDG_METRIC_CONFIG'] = str(folder/'metric_config.json')
    g,d,proof = load(source())
    for p in [g._xyz,g._opacity,g._scaling,g._rotation,g._features_dc,g._features_rest]: p.requires_grad_(True)
    d.deform.requires_grad_(True); d.grid4d.requires_grad_(True); g.velocity_net.requires_grad_(True)
    pool = pool_for(g); known = pool.ids
    with torch.no_grad():
        previous = None
        for t in [0., .25, .5, 1.]:
            geom = geometry(g,warp_at(d,g.get_xyz.detach(),t))
            this = [v[known].clone() for v in geom]
            if previous is not None: assert all(torch.equal(a,b) for a,b in zip(previous,this))
            previous = this
    meta=engine.read_json(DATA/'transforms_train.json')
    f=next(f for f in meta['frames'] if f['file_path'].endswith('0086'))
    w=warp_at(d,g.get_xyz.detach(),85/199)
    pkg=render_call(camera(f,meta,1600),g,w)
    assert pkg['render'].isfinite().all() and pkg['alpha'].isfinite().all()
    out=ROOT/'output/static_pool';out.mkdir(parents=True,exist_ok=True)
    Image.fromarray((pkg['render'].detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(out/'initial_static_pool_t86.png')
    with Image.open(DATA/(f['file_path']+'.png')) as im:
        a=np.asarray(im.convert('RGBA'),dtype=np.float32)/255
    target=torch.tensor(a[:,:,:3]*a[:,:,3:4],device='cuda').permute(2,0,1)[None]
    target=torch.nn.functional.interpolate(target,size=(900,1600),mode='bilinear',align_corners=False)[0]
    loss=(pkg['render']-target).square().mean()
    grads=torch.autograd.grad(loss,[w['d_xyz'],w['d_rotation'],w['d_scaling']],retain_graph=True)
    for grad in grads:
        assert grad.isfinite().all() and torch.equal(grad[known],torch.zeros_like(grad[known]))
    assert sum(float(v[pool.dynamic].abs().sum()) for v in grads)>0
    parameters=[g._xyz,g._opacity,g._scaling,g._rotation,g._features_dc]
    gg=torch.autograd.grad(loss,parameters,retain_graph=True)
    assert all(v.isfinite().all() for v in gg)
    gradient_l1={k:float(v[known].abs().sum()) for k,v in zip(['canonical_xyz','opacity','scale','rotation','SH_DC'],gg)}
    assert all(x>0 for x in gradient_l1.values())
    before=[p.detach().clone() for p in parameters]
    optimizer=torch.optim.Adam(parameters,lr=1e-6);optimizer.zero_grad();loss.backward();optimizer.step()
    changes={k:float((p.detach()[known]-old[known]).abs().sum()) for k,p,old in zip(gradient_l1,parameters,before)}
    assert all(v>0 for v in changes.values())
    # Exercise the actual forked CMR function, recording row identity through a
    # unique time coordinate. These normalized times are diagnostic inputs only.
    from train_pidg import cauchy_momentum_loss_sampled
    coords=torch.cat([g.get_xyz.detach(),torch.linspace(0,1,len(g.get_xyz),device='cuda')[:,None]],1).requires_grad_(True)
    seen=[];original=g.velocity_net.forward
    def record(x,indices=None):
        rows=(x[:,3].detach()*(len(g.get_xyz)-1)).round().long()
        assert not bool(torch.isin(rows,known).any())
        if indices is not None: assert torch.equal(indices,g.particle_indices[rows])
        seen.extend(rows.cpu().tolist())
        return original(x,indices)
    g.velocity_net.forward=record
    try:
        physical=cauchy_momentum_loss_sampled(g,coords,1e-3,1.,sample_size=128,chunk_size=64)
        assert physical.isfinite()
        pg=torch.autograd.grad(physical,list(g.velocity_net.parameters()),allow_unused=True)
        assert all(v.isfinite().all() for v in pg if v is not None)
        assert sum(float(v.abs().sum()) for v in pg if v is not None)>0
    finally:g.velocity_net.forward=original
    assert len(physics_candidates(g))==len(g.get_xyz)-len(known) and seen
    assert engine.sha256(source()/'checkpoints/resume.pt')==proof['sha256']
    engine.write_json(REPORT/'optimizer_probe.json',{'status':'passed','source':proof,'static_rows':len(known),
        'dynamic_rows':len(pool.dynamic),'native_resolution':[1600,900],'static_deformation_output_gradients_zero':True,
        'dynamic_deformation_gradients_nonzero':True,'static_geometry_and_appearance_gradient_L1':gradient_l1,
        'actual_static_parameter_L1_updates':changes,'time_invariant_center_and_covariance':True,
        'actual_CMR_excludes_static_rows':True,'CMR_probe_calls_recorded':len(seen),
        'CMR_loss':float(physical.detach()),'source_unchanged':True})
    print('Native gradient/update and actual dynamic-domain CMR probe passed',flush=True)


if __name__=='__main__':main()
