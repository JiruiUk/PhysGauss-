"""Evaluator only: compare RGB-derived point admission against analytic static GT."""
import numpy as np
from scipy.spatial import cKDTree
from ..prepare_data import ROOT, read, write, sha
from ..adaptive_static.prepare import world_points
from .infer import REPORT,RUNS


def score(xyz,reference,thin,tolerance):
    if len(xyz):
        distance = cKDTree(reference).query(xyz)[0]
        coverage = cKDTree(xyz).query(reference)[0]
        thin_distance = cKDTree(xyz).query(thin)[0]
    else:
        distance = np.zeros(0); coverage = np.full(len(reference),np.inf); thin_distance = np.full(len(thin),np.inf)
    return {'points':len(xyz),'surface_precision':float(np.mean(distance <= tolerance)) if len(xyz) else None,
        'surface_coverage':float(np.mean(coverage <= tolerance)),
        'thin_surface_coverage':float(np.mean(thin_distance <= tolerance)),
        'median_point_to_surface_distance':float(np.median(distance)) if len(xyz) else None,
        'mean_clipped_reference_distance':float(np.minimum(coverage,.5).mean())}


def main():
    assert not (REPORT/'evaluation.json').exists(), 'Preserve completed audit'
    protocol = read(REPORT/'protocol.json'); results = {}
    for case in ['diffuse','glossy']:
        path = RUNS/case/'reconstruction.npz'
        assert sha(path) == read(REPORT/f'{case}_inference.json')['reconstruction_sha256']
        private = ROOT/f'data/derived/clean_static_synthetic/v2/{case}/evaluation_only'
        meta = read(private/'cameras.json'); k = np.asarray(meta['K']); reference,thin = [],[]
        # Each static camera is identical across time; sample once, then spatially voxelize.
        for row in meta['frames']:
            if row['time_index'] != 0: continue
            with np.load(private/'gt'/f"c{row['camera']:02d}_t0.npz") as data:
                depth = data['static_depth']; yy,xx = np.nonzero(depth > 0)
                xyz = world_points(np.c_[xx,yy],depth[yy,xx],np.asarray(row['world_to_camera']),k)
                reference.append(xyz); thin.append(xyz[data['thin_static_visible'][yy,xx]])
        reference,thin = np.concatenate(reference),np.concatenate(thin)
        def unique_surface(points):
            _,indices = np.unique(np.floor(points/.015).astype(int),axis=0,return_index=True)
            return points[indices]
        reference,thin = unique_surface(reference),unique_surface(thin)
        with np.load(path) as data: xyz,rejected = data['xyz'].copy(),data['quarantine'].copy()
        results[case] = {'unfiltered':score(xyz,reference,thin,protocol['evaluation_distance_tolerance']),
                        'quarantine':score(xyz[~rejected],reference,thin,protocol['evaluation_distance_tolerance']),
                        'reference_voxels':len(reference),'thin_reference_voxels':len(thin)}
    write(REPORT/'evaluation.json',{'independent_analytic_static_GT':True,'GT_never_loaded_by_RGB_inference':True,
        'complete_Gaussian_training':False,'point_admission_audit_only':True,
        'coverage_contract':'Recall of camera-visible static surface samples voxelized at0.015; tolerance0.05 scene units. This is not recall of entirely invisible surfaces.',
        'results':results,'inference_manifest_sha256':sha(REPORT/'inference_inputs.json')})
    write(RUNS/'state.json',{'status':'independent_admission_audit_complete'})
    print('SYNTHETIC_INDEPENDENT_GT_AUDIT',results,flush=True)


if __name__ == '__main__': main()
