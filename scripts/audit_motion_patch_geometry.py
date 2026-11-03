"""Model-conditioned local geometry diagnosis. No images or identity labels."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import torch
from variants.staged_reconstruction.prepare_data import ROOT,RUNS,REPORT,read,write,sha


def spectrum(points,index):
    patches=points[index]
    centered=patches-patches.mean(1,keepdims=True)
    covariance=np.einsum('nki,nkj->nij',centered,centered)/patches.shape[1]
    return np.linalg.eigvalsh(covariance).clip(1e-12)


def quantile(value):
    return {str(q):float(np.quantile(value,q)) for q in [.05,.1,.5,.9,.95]}


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--include-I5',action='store_true'); args=parser.parse_args()
    torch.set_num_threads(4)
    from variants.staged_reconstruction.early_depth.train import load_initial
    renderer,scene,opt,initial,base=load_initial()
    del renderer,opt
    x0=initial['model']['pools.0._xyz'].numpy()
    distance,index=cKDTree(x0).query(x0,k=33,workers=1)
    valid=np.flatnonzero(distance[:,-1]<=.08)
    index=index[valid]; reference=spectrum(x0,index)
    paths={'S3U_final6k':RUNS/'S3_union/physics/formal/checkpoint.pt',
           'I1':RUNS/'motion_integrity/formal/I1_curve_isometry/checkpoint.pt',
           'I3':RUNS/'ordinal_depth/formal/checkpoint.pt',
           'I4_rejected_warmup16k':RUNS/'early_depth/formal/ordinal/checkpoint.pt'}
    if args.include_I5: paths['I5']=RUNS/'local_rigidity/formal/rigid/checkpoint.pt'
    target=REPORT/f'motion_patch_geometry_{"with_I5" if args.include_I5 else "before_I5"}.json'
    assert not target.exists(),'Preserve diagnostics'
    times=[2,27,43,62,86,108,146,172,200]; results={}
    for name,path in paths.items():
        digest=sha(path); state=torch.load(path,map_location='cpu',weights_only=False)
        assert digest==read(path.with_name('checkpoint.json'))['sha256']
        if name=='I5': assert state['iteration']==22000
        scene.load_state_dict(state['model'])
        canonical=scene.pools[0].get_xyz.cpu().numpy(); cs=spectrum(canonical,index)
        rows=[]
        for number in times:
            position=scene.dynamic_at((number-1)/199).get_xyz.cpu().numpy()
            ps=spectrum(position,index)
            # Affine singular values are not estimated; these are covariance
            # shape/scale proxies on the same fixed initial neighborhood rows.
            rows.append({'frame':number,
                         'thickness_ratio_to_canonical':quantile(np.sqrt(ps[:,0]/cs[:,0])),
                         'radius_ratio_to_canonical':quantile(np.sqrt(ps.sum(-1)/cs.sum(-1))),
                         'minimum_to_maximum_eigenvalue':quantile(ps[:,0]/ps[:,2]),
                         'fraction_thickness_below_quarter':float(np.mean(ps[:,0]<.0625*cs[:,0])),
                         'fraction_radius_above_double':float(np.mean(ps.sum(-1)>4*cs.sum(-1)))})
        results[name]={'checkpoint_sha256':digest,'iteration':state['iteration'],
                       'canonical_thickness_ratio_to_initial':quantile(np.sqrt(cs[:,0]/reference[:,0])),
                       'canonical_minimum_to_maximum_eigenvalue':quantile(cs[:,0]/cs[:,2]),'times':rows}
        assert sha(path)==digest
        print('PATCH_GEOMETRY',name,'canonical_thickness_median',results[name]['canonical_thickness_ratio_to_initial']['0.5'],
              'world_thickness_medians',[round(r['thickness_ratio_to_canonical']['0.5'],3) for r in rows],flush=True)
    write(target,{'models':results,'fixed_patch_nodes':len(index),'neighbors_including_center':33,'initial_radius_max':.08,
                  'same_initial_neighbor_rows_across_models':True,'initial_checkpoint_sha256':sha(RUNS/'S3_union/warmup/formal/initial.pt'),
                  'no_RGB_read':True,'no_model_updates':True,'material_association_GT':False,
                  'limitation':'Covariance proxies depend on proposed initial neighborhoods. They do not prove object completeness, true geometry or causal failure.'})


if __name__=='__main__': main()
