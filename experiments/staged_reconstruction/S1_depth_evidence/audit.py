"""Read-only evaluator of existing RGB-derived depths; no training or GT selection."""
from pathlib import Path
import json
import numpy as np
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha
from variants.staged_reconstruction.synthetic_audit.input import load
from variants.staged_reconstruction.adaptive_static.prepare import checked_maps,world_points
from variants.staged_reconstruction.nuisance_static.prepare import verify

REPORT = ROOT/'experiments/staged_reconstruction/S1_depth_evidence'
SOURCE = ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2'
CACHE = ROOT/'runs/pidg_staged_reconstruction/S1_clean_visibility/synthetic_v2'


def describe(x):
    x = np.asarray(x)
    return {'n':int(x.size),'mean':float(x.mean()) if x.size else None,
        'median':float(np.median(x)) if x.size else None,
        'p90':float(np.quantile(x,.9)) if x.size else None}


def normal_values(depth,view,k,relative):
    height,width = depth.shape; yy,xx = np.mgrid[:height,:width]
    xyz = world_points(np.c_[xx.ravel(),yy.ravel()],depth.ravel(),view,k).reshape(height,width,3)
    normal = np.cross(xyz[1:-1,2:]-xyz[1:-1,:-2],xyz[2:,1:-1]-xyz[:-2,1:-1])
    length = np.linalg.norm(normal,axis=2); normal /= np.maximum(length[...,None],1e-12)
    center = depth[1:-1,1:-1]; valid = (center > 0)&(length > 1e-8)
    for ay,ax in [(slice(1,-1),slice(2,None)),(slice(1,-1),slice(None,-2)),
            (slice(2,None),slice(1,-1)),(slice(None,-2),slice(1,-1))]:
        valid &= (depth[ay,ax] > 0)&(np.abs(depth[ay,ax]-center) <= relative*center)
    return normal,valid


def main():
    assert not (REPORT/'results.json').exists(), 'Preserve completed diagnostic'
    verify(); protocol = read(SOURCE/'protocol.json'); results,inputs = {},{}
    dataset = read(SOURCE/'dataset.json')
    for p,h in dataset['files'].items(): assert sha(ROOT/p) == h,p
    for case in ['diffuse','glossy']:
        directory = ROOT/f'data/derived/clean_static_synthetic/v2/{case}'
        values = load(directory/'train'); metadata = read(directory/'train/cameras.json')
        receipt = read(SOURCE/f'{case}_inference.json'); maps = []
        for i in range(96):
            path = CACHE/case/f'raw_{i:03d}.npz'; inputs[path.relative_to(ROOT).as_posix()] = sha(path)
            with np.load(path) as data: maps.append(data['depth'].copy())
        ids = np.rint(values['times']*200).astype(int).tolist()
        checked,_,records = checked_maps(maps,receipt['sources'],values['views'],values['K'],ids,protocol)
        assert records == receipt['per_view'], 'Original data-only consistency replay differs'
        # Freeze data-only candidates before reading any evaluator labels.
        rows,errors,visible_errors,thin_errors,normals = [],[],[],[],[]
        totals = {key:0 for key in ['accepted','static_pixels','visible_static_pixels','thin_pixels',
            'accepted_on_static','accepted_visible_static','accepted_thin','same_ray_005','same_ray_002',
            'no_static_surface','in_front_of_static_005','behind_static_005','accepted_under_occluder',
            'normal_eligible','normal_single_GT_face']}
        for i,(depth,row) in enumerate(zip(checked,metadata['frames'])):
            accepted = depth > 0; normal,normal_good = normal_values(depth,values['views'][i],values['K'],protocol['depth_relative_tolerance'])
            path = directory/'evaluation_only/gt'/(Path(row['image']).stem+'.npz')
            with np.load(path) as gt:
                target = gt['static_depth']; surface = target > 0; visible = gt['observed_static_visible']&surface
                thin = gt['thin_static_visible']&surface; on = accepted&surface
                error = np.abs(depth-target); delta = depth-target
                counts = {'accepted':int(accepted.sum()),'static_pixels':int(surface.sum()),
                    'visible_static_pixels':int(visible.sum()),'thin_pixels':int(thin.sum()),
                    'accepted_on_static':int(on.sum()),'accepted_visible_static':int((accepted&visible).sum()),
                    'accepted_thin':int((accepted&thin).sum()),
                    'same_ray_005':int((on&(error <= .05)).sum()),'same_ray_002':int((on&(error <= .02)).sum()),
                    'no_static_surface':int((accepted&~surface).sum()),
                    'in_front_of_static_005':int((on&(delta < -.05)).sum()),
                    'behind_static_005':int((on&(delta > .05)).sum()),
                    'accepted_under_occluder':int((on&~visible).sum()),'normal_eligible':int(normal_good.sum())}
                gt_normal = gt['static_normal']; face_good = surface[1:-1,1:-1].copy()
                for ay,ax in [(slice(1,-1),slice(2,None)),(slice(1,-1),slice(None,-2)),
                        (slice(2,None),slice(1,-1)),(slice(None,-2),slice(1,-1))]:
                    face_good &= surface[ay,ax]&((gt_normal[1:-1,1:-1]*gt_normal[ay,ax]).sum(2) > .99)
                valid = normal_good&face_good
                angle = np.degrees(np.arccos(np.clip(np.abs((normal*gt_normal[1:-1,1:-1]).sum(2)),0,1)))
                normals.append(angle[valid]); counts['normal_single_GT_face'] = int(valid.sum())
                errors.append(error[on]); visible_errors.append(error[accepted&visible]); thin_errors.append(error[accepted&thin])
                for key,value in counts.items(): totals[key] += value
                rows.append({'image':row['image'],'time':row['time'],**counts})
        fraction = lambda a,b:totals[a]/max(totals[b],1)
        results[case] = {'counts':totals,'ratios':{
            'same_ray_static_agreement_005_among_accepted':fraction('same_ray_005','accepted'),
            'same_ray_static_agreement_002_among_accepted':fraction('same_ray_002','accepted'),
            'accepted_on_no_static_surface':fraction('no_static_surface','accepted'),
            'accepted_visible_static_pixel_coverage':fraction('accepted_visible_static','visible_static_pixels'),
            'accepted_thin_pixel_coverage':fraction('accepted_thin','thin_pixels'),
            'normal_eligible_per_accepted_depth':fraction('normal_eligible','accepted')},
            'depth_error_on_static':describe(np.concatenate(errors)),
            'depth_error_on_visible_static':describe(np.concatenate(visible_errors)),
            'depth_error_on_thin':describe(np.concatenate(thin_errors)),
            'unsigned_normal_error_single_GT_face':describe(np.concatenate(normals)),
            'time_groups':{str(t):{k:sum(r[k] for r in rows if r['time'] == t) for k in totals}
                for t in sorted(set(values['times'].tolist()))},'frames':rows}
        print('DEPTH_EVIDENCE',case,json.dumps({k:v for k,v in results[case].items() if k not in ['frames','time_groups']}),flush=True)
    real_path = ROOT/'runs/pidg_staged_reconstruction/S1_geometry_growth/revision2/verified_depths.npz'
    real_receipt = read(ROOT/'experiments/staged_reconstruction/S1_geometry_growth/revision2/discovery.json')
    assert sha(real_path) == real_receipt['artifacts']['verified_depths.npz']
    inputs[real_path.relative_to(ROOT).as_posix()] = sha(real_path)
    with np.load(real_path) as depths:
        real = [{'frame':int(key),'positive':int((depths[key] > 0).sum()),'pixels':int(depths[key].size)} for key in depths.files]
    real = {'inferred_proxy_only':True,'independent_geometry_GT':False,'frames':real,
        'time_thirds':{str(t):{'views':sum(min((r['frame']-1)*3//200,2) == t for r in real),
            'positive_pixels':sum(r['positive'] for r in real if min((r['frame']-1)*3//200,2) == t),
            'pixels':sum(r['pixels'] for r in real if min((r['frame']-1)*3//200,2) == t)} for t in range(3)}}
    for p,h in inputs.items(): assert sha(ROOT/p) == h,p
    verify(); write(REPORT/'results.json',{'diagnostic_only':True,'training_started':False,'GT_selection':False,
        'candidate_generation_before_GT':True,'inherited_depth_consistency_replay_exact':True,
        'distance_scales_are_diagnostics_not_new_promotion_gate':[.02,.05],
        'normal_eligibility':'All four neighboring checked depths positive and relative continuity<=inherited0.025; GT face restriction is evaluation only',
        'synthetic':results,'real':real,'raw_depth_and_real_input_sha256':inputs,
        'dataset_hash_receipt':sha(SOURCE/'dataset.json'),'source_sha256':sha(Path(__file__))})
    print('DEPTH_EVIDENCE_AUDIT_COMPLETE',flush=True)


if __name__ == '__main__': main()
