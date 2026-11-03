"""Independent full-state/sampling/internal-objective audit for S3--S6."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from variants.staged_reconstruction.prepare_data import REPORT,RUNS,DATA,read,write,sha
from variants.staged_reconstruction.train_static import setup,camera
from variants.staged_reconstruction.stage4.bind import load_physics_state
from variants.staged_reconstruction.recovery.coverage import project
from variants.staged_reconstruction.stage5.nonpenetration import sample_sdf,nonpenetration


def sampling(parent,final,updates,frames):
    rng=random.Random();rng.setstate(parent['rng_python']);stack=list(parent['camera_stack']);sequence=[]
    for _ in range(updates):
        if not stack:stack=rng.sample(range(len(frames)),len(frames))
        sequence.append(int(Path(frames[stack.pop()]['file_path']).stem))
    assert stack==final['camera_stack'] and rng.getstate()==final['rng_python']
    return {'exact_python_RNG_and_final_camera_stack':True,'updates_replayed':updates,
        'sampled_frame_ids':sorted(set(sequence)),'sequence_sha256':hashlib.sha256(json.dumps(sequence).encode()).hexdigest(),
        'sample_counts':{str(n):sequence.count(n) for n in sorted(set(sequence))}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['S3','S4','S5','S6'],required=True)
    p.add_argument('--output-name',default='state_objective_audit.json');a=p.parse_args()
    if Path(a.output_name).name!=a.output_name or not a.output_name.endswith('.json'):raise ValueError('Expected a report-local JSON filename')
    stage=a.stage;report=REPORT/stage;path=RUNS/stage/'formal/checkpoint.pt';before=sha(path)
    protocol=read(report/'protocol.json');assert before==read(path.with_name('checkpoint.json'))['sha256']
    from variants.staged_reconstruction.state_recovery.train import freeze as verify_physics
    verify_physics(read(REPORT/'S3/protocol.json'))
    if stage!='S3':
        from variants.staged_reconstruction.continuation import freeze as verify_later
        verify_later(stage)
    renderer=setup();torch.set_num_threads(4);scene,state,bootstrap=load_physics_state(path)
    assert state['iteration']==protocol['updates'] and state['protocol_sha256']==sha(report/'protocol.json')
    parent_path=RUNS/'S2_geometry_recovery/S2_objects/checkpoint.pt' if stage=='S3' else RUNS/f'S{int(stage[1:])-1}/formal/checkpoint.pt'
    assert sha(parent_path)==state['parent_sha256']==protocol['parent_sha256']
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    metadata=read(DATA/'transforms_train.json');frames=metadata['frames'];byid={int(Path(f['file_path']).stem):f for f in frames}
    sampler=sampling(parent,state,protocol['updates'],frames)
    static=torch.load(RUNS/'S1_static/checkpoint.pt',map_location='cpu',weights_only=False)['model']
    static_geometry_equal=all(torch.equal(v,state['model']['static.'+k]) for k,v in static.items() if not k.startswith('_features'))
    static_all_equal=all(torch.equal(v,state['model']['static.'+k]) for k,v in static.items())
    assert static_geometry_equal and (static_all_equal or stage=='S6')
    assert all(v.isfinite().all() for v in state['model'].values())
    assert all(v.isfinite().all() for s in state['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v))
    assert 'physics_graphs' in state
    scales=[]
    with torch.no_grad():
        for t in [0.,85/199,187/199]:
            for k in range(len(scene.objects)):
                value=scene.object_at(k,t).get_scaling
                scales.append({'object':k,'time':t,'min':float(value.min()),'median':float(value.median()),'max':float(value.max())})
    diagnostics=[]
    for n in [2,86,188]:
        try:
            with torch.enable_grad():terms,rows=scene.physics.loss(scene,(n-1)/199,9000,samples=16,h=1/199)
            diagnostics.append({'frame':n,'terms':{k:float(v.detach()) for k,v in terms.items()},'objects':rows,'valid':True})
        except ValueError as error:
            diagnostics.append({'frame':n,'valid':False,'error':str(error),'not_zero_residual':True})
    tracks=[];interior=None
    bindings_file=REPORT/'S4/bindings.json'
    if bindings_file.exists():
        from variants.flow_comparison.temporal import position
        binding=read(bindings_file);all_points=[];all_errors=[]
        assert binding['topology']==[len(o.get_xyz) for o in scene.objects]
        with torch.no_grad():
            for item in binding['bindings']:
                errors=[];k=item['local_early_object'];row=item['local_row']
                for n,uv in zip(item['frames'],item['uv_400']):
                    x=position(scene.deforms[k],scene.objects[k].get_xyz[row:row+1],(n-1)/199)+scene.centers[k]
                    xy,valid=project(x,camera(byid[n],metadata,400))
                    if not bool(valid.all()):raise RuntimeError('Qualified trajectory is behind camera')
                    errors.append(float((xy-xy.new_tensor(uv)).norm()));all_points.append(x)
                tracks.append({'track_id':item['track_id'],'object':k,'mean_reprojection_px400':float(np.mean(errors)),
                    'binding_mean_reprojection_px400':item['binding_mean_reprojection_px'],'observations':len(errors)})
                all_errors.extend(errors)
            if all_points:
                receipt=read(REPORT/'S5/static_volume_receipt.json');volume_path=ROOT/receipt['file'];assert sha(volume_path)==receipt['sha256']
                with np.load(volume_path,allow_pickle=False) as v:
                    points=torch.cat(all_points)
                    sdf=sample_sdf(points,torch.tensor(v['sdf'],device='cuda'),v['origin'],float(v['spacing']))
                    outside=((points<points.new_tensor(v['origin']))|(points>points.new_tensor(v['origin'])+(points.new_tensor(v['sdf'].shape)-1)*float(v['spacing']))).any(-1)
                _,interior=nonpenetration(sdf,torch.ones(len(sdf),device='cuda',dtype=torch.bool),receipt['discretization_margin'],True)
                interior['signed_distance_min']=float(sdf.min());interior['margin']=receipt['discretization_margin']
                interior['outside_volume_observations']=int(outside.sum())
    numerical=read(report/f'{stage}_val_{protocol["updates"]:05d}.json')['metrics']
    test=read(report/f'{stage}_test_{protocol["updates"]:05d}.json')['metrics']
    gates={'dynamic_PSNR':numerical['dynamic_PSNR']>=20.,'dynamic_only_alpha':numerical['dynamic_alpha']>=.85,
        'static_preservation_from_S1':35.643816-numerical['static_PSNR']<=1.}
    activation=None
    if stage in ['S5','S6']:
        lines=[json.loads(s) for s in (RUNS/stage/'formal/objectives.jsonl').read_text(encoding='utf8').splitlines()]
        values=[x for x in lines if 'conditional_interior' in x]
        assert len(values)==protocol['updates']//protocol['tracks_every']
        activation={'objective_calls':len(values),'eligible_observation_instances':sum(x['conditional_interior']['eligible_count'] for x in values),
            'active_violation_instances':sum(x['conditional_interior']['active_violations'] for x in values),
            'calls_with_any_active_violation':sum(x['conditional_interior']['active_violations']>0 for x in values),
            'maximum_proxy_penetration_during_training':max(x['conditional_interior']['max_proxy_penetration'] or 0. for x in values),
            'nonzero_prior_objective_calls':sum(x['nonpenetration_loss']>0 for x in values),
            'all_recorded_prior_values_finite':all(np.isfinite(x['nonpenetration_loss']) for x in values),
            'not_collision_ground_truth':True}
    assert sha(path)==before
    write(report/a.output_name,{'stage':stage,'checkpoint_sha256':before,'iteration':state['iteration'],
        'parent_sha256':sha(parent_path),'numeric_gates':gates,'native_validation':numerical,'native_test':test,
        'static_geometry_exact_S1':static_geometry_equal,'static_all_exact_S1':static_all_equal,'finite_model_and_optimizer':True,
        'optimizer_state_count':len(state['optimizer']['state']),'sampler':sampler,'scale_diagnostics':scales,
        'fixed_frame_internal_physics':diagnostics,'all_fixed_frame_physics_valid':all(x['valid'] for x in diagnostics),
        'qualified_tracks':tracks,'track_observation_mean_px400':float(np.mean(all_errors)) if tracks else None,
        'conditional_interior':interior,'training_conditional_interior':activation,'manual_visual_review_pending':True,
        'limits':'Training-image correspondences, model stress/velocity consistency and conditional SDF are proxies; no independent 3D/material/force GT; late physical ID remains unknown'})
    print('STAGE_STATE_OBJECTIVE_AUDIT',stage,gates,flush=True)


if __name__=='__main__':main()
