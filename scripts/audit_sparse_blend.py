"""Read-only LBS rotation/contraction diagnosis; no training loss or physics module."""
import numpy as np
import torch
from variants.staged_reconstruction.prepare_data import ROOT,RUNS,REPORT,read,write,sha
from variants.staged_reconstruction.sparse_motion.core import weights,matrix,interpolate


def differential(query,controls,radius,node):
    ids,w=weights(query,controls,radius)
    rotations=matrix(node['d_xyz'][ids,:4])
    moved=(rotations@query[:,None,:,None]).squeeze(-1)+node['d_xyz'][ids,4:]
    blend=(w[:,:,None,None]*rotations).sum(1)
    log_gradient=-(query[:,None]-controls[ids])/radius[ids].square().clamp_min(1e-8)[...,None]
    weight_gradient=w[:,:,None]*(log_gradient-(w[:,:,None]*log_gradient).sum(1,keepdim=True))
    mean=(w[:,:,None]*moved).sum(1,keepdim=True)
    full=blend+torch.einsum('nki,nkj->nij',moved-mean,weight_gradient)
    return blend,full


def quantiles(value):
    return {str(q):float(torch.quantile(value,q)) for q in [.01,.1,.5,.9,.99]}


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    path=RUNS/'sparse_motion/formal/checkpoint.pt'; digest=sha(path)
    assert digest==read(path.with_name('checkpoint.json'))['sha256']
    from variants.staged_reconstruction.early_depth.train import load_initial
    renderer,scene,opt,initial,base=load_initial(); del renderer,opt
    state=torch.load(path,map_location='cpu',weights_only=False); scene.load_state_dict(state['model'])
    plan=read(REPORT/'sparse_motion/protocol.json'); control_path=ROOT/plan['controls']
    assert sha(control_path)==plan['controls_sha256']
    with np.load(control_path) as value:
        controls=torch.tensor(value['xyz'],device='cuda'); radius=torch.tensor(value['radius'],device='cuda')
    x=scene.pools[0].get_xyz; rows=[]
    for number in [2,27,43,62,86,108,146,172,200]:
        t=(number-1)/199
        node=scene.deforms[0].step(controls,controls.new_full((len(controls),1),t))
        blended,full=differential(x,controls,radius,node)
        bs=torch.linalg.svdvals(blended); fs=torch.linalg.svdvals(full); determinant=torch.linalg.det(full)
        rows.append({'frame':number,'rotation_blend_min_singular_value':quantiles(bs[:,-1]),
                     'full_map_min_singular_value':quantiles(fs[:,-1]),'full_map_max_singular_value':quantiles(fs[:,0]),
                     'fraction_rotation_blend_below_quarter':float((bs[:,-1]<.25).float().mean()),
                     'fraction_full_map_below_quarter':float((fs[:,-1]<.25).float().mean()),
                     'fraction_full_map_negative_determinant':float((determinant<0).float().mean())})
        print('SPARSE_BLEND',number,'rotation_q10',rows[-1]['rotation_blend_min_singular_value']['0.1'],
              'negative_det',rows[-1]['fraction_full_map_negative_determinant'],flush=True)
    assert sha(path)==digest
    output=REPORT/'sparse_motion/blend_diagnostic.json'; assert not output.exists()
    write(output,{'checkpoint_sha256':digest,'controls_sha256':sha(control_path),'rows':rows,
        'local_derivative_includes_RBF_weight_gradient':True,'no_RGB_read':True,'optimizer_updates':0,
        'not_constitutive_F_estimator_or_training_branch':True,
        'limitations':'Piecewise local derivative away from KNN support changes. Model-conditioned, not true deformation or material correspondence.'})


if __name__=='__main__': main()
