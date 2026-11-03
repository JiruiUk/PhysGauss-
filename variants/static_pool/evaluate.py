"""Fixed final-state evaluation with native rendering and no pseudo novel-view GT."""
from pathlib import Path
import importlib.util
import os
import sys
import numpy as np
import torch
from .run import ROOT, REPORT, RUNS, DATA, NAMES, source, plan, engine, freeze
from .prepare import prepare

OUT = ROOT/'output/static_pool'


def measurement():
    spec=importlib.util.spec_from_file_location('r0_existing_measurement',ROOT/'experiments/transfer_integrity/evaluate.py')
    e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
    # The old helper inserts its own historical tree. Select the new renderer
    # explicitly before any scene/renderer import; all inherited math is identical.
    repo=prepare();sys.path.insert(0,str(repo))
    e.r.OUT=OUT;e.r.REPORT=REPORT
    e.r.MODELS={'S0_previous_control':ROOT/'runs/pidg_scratch_integrity/pilot/S0_scratch_control/rubber_duck',
               **{n:RUNS/'pilot'/n/'rubber_duck' for n in NAMES}}
    import gaussian_renderer
    assert Path(gaussian_renderer.__file__).resolve()==(repo/'gaussian_renderer/__init__.py').resolve()
    return e


def proof(e):
    import json
    freeze(prepare());states={};samples=[]
    for name in NAMES:
        directory=e.r.MODELS[name]
        rows=[json.loads(x) for x in (directory/'camera_sampling.jsonl').read_text().splitlines()]
        assert [z['iteration'] for z in rows]==list(range(15001,50001));samples.append(rows)
        g,d,p=e.select(directory);assert p['iteration']==50000 and p['point_count']==88916
        states[name]=p;del g,d;torch.cuda.empty_cache()
    assert samples[0]==samples[1]
    cs=[engine.read_json(e.r.MODELS[n]/'metric_config.json') for n in NAMES]
    assert {k for k in cs[0] if cs[0][k]!=cs[1][k]}=={'hard_static_pool'}
    engine.write_json(REPORT/'training_verification.json',{'states':states,'actual_updates_per_arm':35000,
        'same_camera_samples':35000,'only_main_flag_differs':True,'source_sha256':plan()['source_checkpoint_sha256'],
        'historical_S0_reference_only':True})


@torch.no_grad()
def static_diagnostics(e):
    from .probe import geometry
    from variants.kinematic_tuning.model import center_trajectory
    from variants.flow_comparison.temporal import warp_at
    from output.short_clip_diagnostic.camera_math import w2c,intrinsic,project
    old=engine.read_json(ROOT/plan()['binding_metadata'])
    with np.load(old['path'],allow_pickle=False) as z:
        static_ids=z['static_ids'].copy();anchor_ids=z['anchor_ids'].copy();anchor_xyz=z['anchor_xyz'].copy()
    audit=engine.read_json(ROOT/'experiments/transfer_integrity/static_image_audit.json')
    xyz=np.asarray([t['xyz'] for t in audit['tracks']]);by_frame={}
    for row,x in zip(anchor_ids,anchor_xyz):
        distance=np.linalg.norm(xyz-x,axis=1);j=int(distance.argmin());assert distance[j]<1e-5
        track=audit['tracks'][j]
        for k,(n,uv) in enumerate(zip(track['frames'],track['uv'])):
            by_frame.setdefault(n,[]).append((int(row),uv,k%2==1))
    meta=engine.read_json(DATA/'transforms_train.json')
    frames={int(Path(f['file_path']).stem):f for f in meta['frames']};k=intrinsic(meta['camera_angle_x'],'legacy')
    output={}
    for name in NAMES:
        g,d,p=e.select(e.r.MODELS[name]);ids=torch.tensor(static_ids,device='cuda')
        rows=[];errors=[];triangulation_held_errors=[];initial=None
        for n in range(1,201):
            centers,cov=geometry(g,warp_at(d,g.get_xyz.detach(),(n-1)/199))
            now=[centers[ids],cov[ids]]
            if initial is None:initial=[v.clone() for v in now]
            rows.append({'frame':n,'center_difference_RMS':float((now[0]-initial[0]).square().mean().sqrt()),
                         'covariance_difference_RMS':float((now[1]-initial[1]).square().mean().sqrt())})
            if n in by_frame:
                items=by_frame[n];world=centers[[x[0] for x in items]].cpu().numpy()
                uv,depth=project(world,w2c(frames[n]['transform_matrix'],'legacy'),k)
                err=np.linalg.norm(uv-np.asarray([x[1] for x in items]),axis=1)
                # Invalid projections count as failures, not missing successes.
                err=np.where(np.isfinite(err)&(depth>.2),err,400.)
                errors.extend(err.tolist());triangulation_held_errors.extend(err[[x[2] for x in items]].tolist())
        result={'full_state':p,'static_rows':len(ids),'times':rows,
            'max_center_time_difference_RMS':max(x['center_difference_RMS'] for x in rows),
            'max_covariance_time_difference_RMS':max(x['covariance_difference_RMS'] for x in rows),
            'static_image_track_reprojection_mean_px_400':float(np.mean(errors)),
            'static_image_track_reprojection_median_px_400':float(np.median(errors)),
            'triangulation_held_observation_mean_px_400':float(np.mean(triangulation_held_errors)),
            'track_observations':len(errors),'images_are_original_training_images':True,
            'held_is_only_for_triangulation_not_model_training':True,'independent_3D_GT':False}
        if name==NAMES[1]:
            assert result['max_center_time_difference_RMS']==result['max_covariance_time_difference_RMS']==0
        output[name]=result;del g,d;torch.cuda.empty_cache()
    engine.write_json(REPORT/'static_diagnostics.json',output)


@torch.no_grad()
def static_visible_support(e):
    from variants.flow_comparison.temporal import warp_at
    from variants.shape_coherence.observations import camera
    from variants.association_training.evaluate import render_call
    from PIL import Image
    binding=engine.read_json(REPORT/'binding.json')
    with np.load(binding['path'],allow_pickle=False) as z:ids=torch.tensor(z['static_ids'],device='cuda')
    output={}
    for name in NAMES:
        g,d,proof=e.select(e.r.MODELS[name]);color=torch.zeros(len(g.get_xyz),3,device='cuda');color[ids]=1
        groups={}
        for split in ['train','val']:
            meta=engine.read_json(ROOT/f'data/derived/kinematic_resplit/transforms_{split}.json');rows=[]
            for f in meta['frames']:
                n=int(Path(f['file_path']).stem)
                with Image.open(ROOT/f'data/PIDG/rubber_duck/train/{n:04d}.png') as im:
                    a=np.array(im.convert('RGBA').resize((400,225)),dtype=float)/255
                rgb=a[:,:,:3]*a[:,:,3:4]
                blue=(rgb[:,:,2]>rgb[:,:,0]*1.3+.02)&(rgb[:,:,2]>rgb[:,:,1]*1.05)&(a[:,:,3]>.95)
                yellow=(np.minimum(rgb[:,:,0],rgb[:,:,1])>rgb[:,:,2]*1.35+.03)&(rgb[:,:,0]>.15)&(rgb[:,:,1]>.13)&(a[:,:,3]>.95)
                # Native full-layer rendering, then area reduction for region sums.
                w=warp_at(d,g.get_xyz.detach(),(n-1)/199);cam=camera(f,meta,1600)
                p=render_call(cam,g,w,color)
                share=torch.nn.functional.interpolate(p['render'].mean(0)[None,None],size=(225,400),mode='area')[0,0].cpu().numpy()
                alpha=torch.nn.functional.interpolate(p['alpha'][None],size=(225,400),mode='area')[0,0].cpu().numpy()
                assert np.isfinite(share).all() and share.min()>-1e-6 and (share-alpha).max()<1e-5
                rows.append({'frame':n,'static_share_of_blue_alpha':float(share[blue].sum()/max(alpha[blue].sum(),1e-10)),
                    'static_mass_on_blue':float(share[blue].sum()),'blue_alpha_mass':float(alpha[blue].sum()),
                    'static_yellow_leakage_fraction':float(share[yellow].sum()/max(share.sum(),1e-10))})
            groups[split]={'frames':rows,'mean_static_share_of_blue_alpha':float(np.mean([r['static_share_of_blue_alpha'] for r in rows])),
                          'mean_static_yellow_leakage_fraction':float(np.mean([r['static_yellow_leakage_fraction'] for r in rows]))}
        output[name]={'full_state':proof,'groups':groups,'region_labels_are_color_proxies':True,'full_compositing_includes_dynamic_occluders':True}
        del g,d;torch.cuda.empty_cache()
    engine.write_json(REPORT/'static_visible_support.json',output)


def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);e=measurement();proof(e)
    eval_files=[Path(__file__),ROOT/'experiments/transfer_integrity/evaluate.py',ROOT/'output/fixed_camera_temporal/render_sequences.py']
    eval_hash={p.relative_to(ROOT).as_posix():engine.sha256(p) for p in eval_files}
    engine.write_json(REPORT/'evaluation_source_lock.json',eval_hash)
    e.observations();static_diagnostics(e);static_visible_support(e)
    sys.argv=[sys.argv[0]];e.r.main();freeze(prepare())
    assert all(engine.sha256(ROOT/p)==h for p,h in eval_hash.items())
    engine.write_json(REPORT/'evaluation_complete.json',{'status':'evaluated_pending_export_and_review','native_renders':2400,
        'validation30_test50_per_model':True,'selected_static_geometry_times':200,'novel_view_GT_available':False})


if __name__=='__main__':main()
