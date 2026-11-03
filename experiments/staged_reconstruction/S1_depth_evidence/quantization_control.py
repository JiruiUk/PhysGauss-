"""Evaluator-only controls for depth-to-normal diagnostics, never a train prior."""
import numpy as np
from pathlib import Path
from audit import REPORT,SOURCE,CACHE,normal_values,describe
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha
from variants.staged_reconstruction.synthetic_audit.input import load
from variants.staged_reconstruction.adaptive_static.prepare import checked_maps
from variants.staged_reconstruction.nuisance_static.prepare import verify


def main():
    assert not (REPORT/'quantization_control.json').exists()
    verify(); audit = read(REPORT/'results.json'); protocol = read(SOURCE/'protocol.json'); results = {}
    for p,h in audit['raw_depth_and_real_input_sha256'].items(): assert sha(ROOT/p) == h,p
    for case in ['diffuse','glossy']:
        directory = ROOT/f'data/derived/clean_static_synthetic/v2/{case}'
        meta = read(directory/'train/cameras.json'); data = load(directory/'train')
        receipt = read(SOURCE/f'{case}_inference.json'); near,far = receipt['camera_derived_depth_range']
        inverse_grid = np.linspace(1/near,1/far,protocol['depth_samples'])
        maps = []
        for i in range(96):
            with np.load(CACHE/case/f'raw_{i:03d}.npz') as d: maps.append(d['depth'].copy())
        checked,_,records = checked_maps(maps,receipt['sources'],data['views'],data['K'],
            np.rint(data['times']*200).astype(int).tolist(),protocol)
        assert records == receipt['per_view']
        angle_exact,angle_quantized,depth_errors,spacings = [],[],[],[]
        for i,row in enumerate(meta['frames']):
            _,eligible = normal_values(checked[i],data['views'][i],data['K'],protocol['depth_relative_tolerance'])
            with np.load(directory/'evaluation_only/gt'/(Path(row['image']).stem+'.npz')) as gt:
                depth,normal = gt['static_depth'],gt['static_normal']; face = (depth[1:-1,1:-1] > 0)
                for ay,ax in [(slice(1,-1),slice(2,None)),(slice(1,-1),slice(None,-2)),
                        (slice(2,None),slice(1,-1)),(slice(None,-2),slice(1,-1))]:
                    face &= (depth[ay,ax] > 0)&((normal[1:-1,1:-1]*normal[ay,ax]).sum(2) > .99)
                fixed = eligible&face
                index = np.rint((1/np.maximum(depth,1e-8)-inverse_grid[0])/(inverse_grid[1]-inverse_grid[0])).astype(int)
                index = np.clip(index,0,len(inverse_grid)-1)
                quantized = np.where(depth > 0,1/inverse_grid[index],0)
                assert np.all(depth[depth > 0] >= near) and np.all(depth[depth > 0] <= far)
                for value,bucket in [(depth,angle_exact),(quantized,angle_quantized)]:
                    n,_ = normal_values(value,data['views'][i],data['K'],protocol['depth_relative_tolerance'])
                    angle = np.degrees(np.arccos(np.clip(np.abs((n*normal[1:-1,1:-1]).sum(2)),0,1)))
                    bucket.append(angle[fixed])
                on = (checked[i] > 0)&(depth > 0)
                depth_errors.append(np.abs(quantized[on]-depth[on]))
                spacings.append(np.abs(1/inverse_grid[np.minimum(index[on]+1,len(inverse_grid)-1)]-1/inverse_grid[index[on]]))
        results[case] = {'same_pixels_as_audited_normals':True,
            'exact_GT_depth_to_normal_error_deg':describe(np.concatenate(angle_exact)),
            'oracle_nearest192_grid_GT_depth_to_normal_error_deg':describe(np.concatenate(angle_quantized)),
            'oracle_nearest192_grid_depth_error':describe(np.concatenate(depth_errors)),
            'local_grid_spacing':describe(np.concatenate(spacings))}
        assert results[case]['exact_GT_depth_to_normal_error_deg']['mean'] < .05,'Coordinate/normal-estimator sanity failed'
    verify(); write(REPORT/'quantization_control.json',{'evaluator_only':True,'oracle_not_reconstructor':True,
        'GT_never_exported_as_training_input':True,'candidate_thresholds_changed':False,'results':results,
        'input_audit_sha256':sha(REPORT/'results.json'),'source_sha256':sha(Path(__file__))})
    print('DEPTH_QUANTIZATION_CONTROL',results,flush=True)


if __name__ == '__main__': main()
