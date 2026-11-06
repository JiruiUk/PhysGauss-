"""One preregistered comparison, serial native qualification and training."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
from variants.staged_reconstruction.i10_surface_graph import runtime as a
from variants.staged_reconstruction.i10_surface_graph.core import build_patches,build_edges,add_constraints
from variants.staged_reconstruction.i10_tracking.io import ROOT,DATA,read,sha
from experiments.staged_reconstruction.RubberDuck_joint_pilot import launch as old


def worker(phase,arm='surface_graph',replicate=0):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=a.REPORT/f'{stamp}_{phase}_{arm}_{replicate}.log';path.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--enable-surface-graph','--phase',phase,'--arm',arm,'--replicate',str(replicate)]
    with path.open('x',encoding='utf-8') as f:code=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
    assert code==0,f'Failed log retained: {path}'
    return dict(log=path.relative_to(ROOT).as_posix(),sha256=sha(path))


def prepare():
    import numpy as np
    import torch
    from variants.staged_reconstruction.i10_tracking.core import project_world
    r=a.install(True);assert not r.stopped() and not (a.REPORT/'source_freeze.json').exists()
    assert read(a.REPORT/'unit_tests.json')['passed'];torch.set_num_threads(4)
    p=read(a.BASE/'protocol.json');e=read(ROOT/p['evidence'])
    assert not read(ROOT/p['evidence_review'])['pending_review']
    with np.load(ROOT/p['trajectories']) as f:
        xyz=f['xyz'];projected,_=project_world(xyz,f['views'],f['k'])
    patches=build_patches(e,xyz,projected,p);edges=build_edges(patches,p)
    assert sum(q['is_anchor'] for q in patches)==122
    r.atomic_json(a.REPORT/'hypotheses.json',dict(patches=patches,edges=edges,
        summary=dict(nodes=len(patches),anchors=sum(q['is_anchor'] for q in patches),
        surface_patches=sum(bool(q.get('members')) for q in patches if q['is_anchor']),
        neighbor_hypotheses=sum(not q['is_anchor'] for q in patches),
        distinct_rows=len({v for q in patches for v in q['rows']}),proposed_edges=len(edges),
        fit_only_membership_and_weights=True,no_physical_identity_or_GT=True)))
    renderer,scene,opts,_,parent=r.load('surface_graph')
    before=deepcopy(r.cpu_values(dict(model=scene.state_dict(),optimizers=[o.state_dict() for o in opts],
        rng=a.rng_state(),camera=parent['camera_stack'],graphs=parent['physics_graphs'])))
    summary=a.refresh(renderer,scene,True)
    r.atomic_json(a.REPORT/'initial_coverage.json',dict(summary=summary,passed=summary['active_edges']>=p['min_initial_active_edges'] and summary['active_neighbor_rows']>=p['min_initial_neighbor_rows']))
    assert read(a.REPORT/'initial_coverage.json')['passed'], summary
    after=r.cpu_values(dict(model=scene.state_dict(),optimizers=[o.state_dict() for o in opts],
        rng=a.rng_state(),camera=parent['camera_stack'],graphs=parent['physics_graphs']))
    r.assert_equivalent(before,after,0,0)
    r.atomic_json(a.REPORT/'initial_health.json',dict(controller=a.STATE,summary=summary,
        full_model_Adam_RNG_camera_physics_unchanged=True,train_only=True))
    files={}
    inherited=read(ROOT/'experiments/staged_reconstruction/RubberDuck_I10_track_neighborhood/visible/v1/source_freeze.json')
    for name,h in inherited.get('files',inherited).items():assert sha(ROOT/name)==h,name;files[name]=h
    paths=list((ROOT/'variants/staged_reconstruction/i10_surface_graph').glob('*.py'))
    paths += [a.BASE/'plan.md',a.BASE/'protocol.json',a.BASE/'preregistration.json',Path(__file__),
        a.REPORT/'hypotheses.json',a.REPORT/'initial_health.json',ROOT/'tests/test_i10_surface_graph.py',ROOT/p['evidence_review']]
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'envs'):paths.append(path)
    for path in paths:files[path.relative_to(ROOT).as_posix()]=sha(path)
    r.atomic_json(a.REPORT/'source_freeze.json',files);r.verify()
    print('L3_NATIVE_INPUTS_AND_SOURCE_FROZEN',len(files),summary,flush=True)


def native():
    import numpy as np
    import torch
    from variants.staged_reconstruction.train_static import camera
    from variants.staged_reconstruction.rubber_joint.core import objective as baseline
    r=a.install(True);r.verify();torch.set_num_threads(4)
    renderer,scene,opts,_,parent=r.load('surface_graph');state_before=deepcopy(r.cpu_values(r.snapshot(scene,opts,0,parent['camera_stack'],[])))
    centers,measures=a.measure(renderer,scene)
    assert measures==[s['measurement'] for s in a.STATE['patches']]
    r.assert_equivalent(state_before,r.snapshot(scene,opts,0,parent['camera_stack'],[]),0,0)
    try:
        with a.evidence_context(scene):
            torch.rand(4,device='cuda');next(scene.parameters()).requires_grad_(False)
            raise ValueError('expected context failure fixture')
    except ValueError:pass
    r.assert_equivalent(state_before,r.snapshot(scene,opts,0,parent['camera_stack'],[]),0,0)
    active=[i for i,s in enumerate(a.STATE['edges']) if s['status']=='active'];assert active,'No evidenced local connection; preserve protocol, report failure'
    edge=a.EDGES[active[0]];index=edge['start'];frame=a.META['frames'][index];cam=camera(frame,a.META,1600);target=r.frame_target(frame)
    base,pkg,_,_=baseline(renderer,scene,None,cam,frame['time'],target)
    disabled,_,_,_=add_constraints(base,*([None]*9),enabled=False);assert disabled is base
    images={k:pkg[k].detach().clone() for k in ['render','alpha','depth']};disabled.backward()
    grads={n:None if p.grad is None else p.grad.detach().clone() for n,p in scene.named_parameters()}
    scene.zero_grad(set_to_none=True);del base,pkg,disabled
    base,pkg,_,_=baseline(renderer,scene,None,cam,frame['time'],target);base.backward()
    r.assert_equivalent(grads,{n:p.grad for n,p in scene.named_parameters()},1e-6,1e-5)
    assert all(torch.equal(images[k],pkg[k]) for k in images)
    scene.zero_grad(set_to_none=True);del base,pkg,grads
    loss,pkg,_,_=a.objective(renderer,scene,None,cam,frame['time'],target)
    assert all(torch.equal(images[k],pkg[k]) for k in images);loss.backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in scene.parameters())
    scene.zero_grad(set_to_none=True);del loss,pkg,images
    world=scene.dynamic_at(frame['time']).get_xyz;perturb=torch.zeros_like(world)
    row=a.PATCHES[edge['j']]['rows'][0];perturb[row,0]=.003
    _,track,local,counts=add_constraints(world.sum()*0,world+perturb,cam.world_view_transform.T,
        world.new_tensor(a.ASSET['k']),index,a.PATCHES,a.EDGES,a.STATE['patches'],a.STATE['edges'],a.CONFIG,True)
    assert counts['edges']>0 and local>0;local.backward();norms={}
    for prefix in ['pools.0._xyz','grids.','networks.']:
        values=[p.grad for n,p in scene.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert values and all(torch.isfinite(v).all() for v in values)
        norms[prefix]=sum(float(v.abs().sum()) for v in values);assert norms[prefix]>0
    assert all(p.grad is None for n,p in scene.named_parameters() if n.startswith(('static.','physics.')))
    # Packed group evidence must equal individual patch renders, with exact alpha/D.
    with torch.no_grad():
        groups=[a.PATCHES[edge['i']]['rows'],a.PATCHES[edge['j']]['rows']]
        packed=a.native(renderer,scene,cam,frame['time'],a.colors_for(scene,groups))
        for channel,group in enumerate(groups):
            single=a.native(renderer,scene,cam,frame['time'],a.colors_for(scene,[group]))
            assert torch.allclose(packed['render'][channel],single['render'][0],atol=1e-7,rtol=1e-6)
            assert all(torch.equal(packed[k],single[k]) for k in ['alpha','depth'])
    r.atomic_json(a.REPORT/'native_check.json',dict(passed=True,width=1600,
        default_off_RGB_alpha_D_exact=True,default_off_gradients_original_tolerance=True,
        all_patch_measurements_independent_process_exact=True,audit_no_mutation_and_exception_rng_flags=True,
        packed_unpacked_surface_contribution=True,post_T_buffer_not_used=True,
        intended_local_gradients=norms,static_physics_frozen=True))
    r.verify();print('L3_NATIVE_QUALIFICATION_PASS',norms,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-surface-graph',action='store_true')
    parser.add_argument('--phase',choices=['prepare','native','qualify','smoke','recovery','train','pipeline'],default='prepare')
    parser.add_argument('--arm',choices=['rgb_control','surface_graph'],default='surface_graph')
    parser.add_argument('--replicate',type=int,choices=[0,1],default=0);args=parser.parse_args()
    if not args.enable_surface_graph:raise SystemExit('Revocable surface default off')
    r=a.install(True);old.worker=worker;assert not r.stopped()
    if args.phase=='prepare':prepare()
    elif args.phase=='native':native()
    elif args.phase=='qualify':old.qualify()
    elif args.phase=='smoke':r.train(args.arm,True)
    elif args.phase=='recovery':r.recovery_worker(args.arm,args.replicate)
    elif args.phase=='train':r.train(args.arm)
    else:old.pipeline()


if __name__=='__main__':main()
