"""Own paths and multi-row evidence over frozen L8 native training machinery."""
import numpy as np
import torch
from ..i10_ordinal_depth import runtime as q
from ..i10_tracking.core import project_world,patch_ncc
from ..i10_track_neighborhood.core import patch_visibility
from ..rubber_visibility.worker import native,colors_for
from ..train_static import camera
from .core import sampled_depth

BASE=q.ROOT/'experiments/staged_reconstruction/RubberDuck_relative_order'
REPORT=BASE/'v1';RUNS=q.ROOT/'runs/pidg_staged_reconstruction/RubberDuck_relative_order/v1'
OUT=q.ROOT/'output/staged_reconstruction/RubberDuck_relative_order/v1'
ORIGINAL_SNAPSHOT=q.snapshot


def __getattr__(name):return getattr(q,name)


def measure(renderer,scene):
    result={}
    with q.evidence_context(scene):
        for index in sorted({p['frame'] for p in q.BANK['fit']['pairs']}):
            assert not q.r.stopped(),'User stop request'
            frame=q.META['frames'][index];cam=camera(frame,q.META,400)
            pairs=[p for p in q.BANK['fit']['pairs'] if p['frame']==index]
            pkg=scene.render(renderer,cam,frame['time']);alpha=pkg['alpha'].squeeze().cpu().numpy()
            observed=q.r.frame_target(frame,400).mean(0).cpu().numpy()*255;pred=pkg['render'].mean(0).cpu().numpy()*255
            world=scene.dynamic_at(frame['time']).get_xyz.cpu().numpy();rows=sorted({p['row'] for p in pairs})
            for begin in range(0,len(rows),3):
                ids=rows[begin:begin+3];packed=native(renderer,scene,cam,frame['time'],colors_for(scene,[[i] for i in ids]))
                assert all(torch.equal(pkg[k],packed[k]) for k in ['alpha','depth'])
                for channel,row in enumerate(ids):
                    weight=packed['render'][channel].cpu().numpy();uv,z=project_world(world[row:row+1],q.ASSET['views'][index:index+1],q.ASSET['k'])
                    for p in pairs:
                        if p['row']!=row:continue
                        point=np.array(p['uv'][0]);error=float(np.linalg.norm(uv[0,0]-point));vis=patch_visibility(weight,alpha,point,q.CONFIG)
                        ncc,x,y=patch_ncc(observed,pred,point[None],point[None],2);depths,alphas=sampled_depth(pkg,p['uv'])
                        known=bool(torch.isfinite(depths).all() and (depths>0).all() and (alphas>q.CONFIG['minimum_rendered_alpha']).all() and vis['known'] and z[0,0]>.01)
                        passed=known and vis['passed'] and error<=q.CONFIG['maximum_projection_error'] and ncc[0]>=q.CONFIG['ncc_min'] and min(x[0],y[0])>=q.CONFIG['patch_std_min']
                        result[p['id']]=dict(known=known,passed=bool(passed),projection_error=error,visibility=vis,ncc=float(ncc[0]),
                            observed_std=float(x[0]),rendered_std=float(y[0]),endpoint_depths=depths.cpu().tolist(),endpoint_alpha=alphas.cpu().tolist())
    return result


def snapshot(scene,opts,step,stack,trace):
    state=ORIGINAL_SNAPSHOT(scene,opts,step,stack,trace)
    state.update(stage='L8R_scale_free_relative_order',metric_calibration_used_for_targets=False,
        first_interval_pair_failure_preserved=True)
    return state


def install(enabled=False):
    if not enabled:return q.r
    q.BASE=BASE;q.REPORT=REPORT;q.RUNS=RUNS;q.OUT=OUT;q.measure=measure;q.snapshot=snapshot
    return q.install(True)
