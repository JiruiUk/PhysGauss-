"""Post-fit conditional physical-parameter Jacobian plus manufactured ambiguity test."""
import argparse
from argparse import Namespace
import json
import os
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pidg_benchmark import read_json,write_json,sha256
from pidg_ext.identifiability import jacobian_diagnostics,ambiguity_experiment


def diagnose(run):
    run=Path(run).resolve();metadata=read_json(run/'reproduction.json')
    repo=ROOT/'variants/constitutive_pidg/code'
    sys.path.insert(0,str(repo))
    os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
    os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
    from scene_PIDG import GaussianModel,DeformModel
    from utils.general_utils import quaternion_to_matrix
    from reproduction.scripts import training_checkpoint as baseline
    from pidg_ext.runtime import Extension
    from pidg_ext.deformation_gradient import canonical_neighbors,estimate_F
    from pidg_ext.losses import predicted_stress
    from pidg_ext.forces import momentum_components,StructuredForce
    from pidg_ext.constitutive import NeoHookean
    state=baseline.load_state(run/'checkpoints/resume.pt')
    dataset=Namespace(**state['dataset']);opt=Namespace(**state['opt'])
    gaussians=GaussianModel(dataset.sh_degree)
    deform=DeformModel(grid_args=dataset.grid_args,net_args=dataset.network_args,scale_xyz=dataset.scale_xyz,
                       reg_spatial_able=False,reg_temporal_able=False)
    deform.train_setting(opt);baseline.restore_models(state,gaussians,deform,opt)
    for optimizer in (gaussians.optimizer,deform.optimizer):
        for group in optimizer.param_groups:
            for parameter in group['params']:parameter.requires_grad_(False)
    frames=read_json(Path(dataset.source_path)/'transforms_train.json')['frames']
    extension=Extension(run/'extension_config.json',run,len(frames))
    extension.restore_checkpoint(state.get('constitutive_extension'),gaussians.get_xyz.device)
    X=gaussians.get_xyz
    ids=torch.arange(0,len(X),max(1,len(X)//64),device=X.device)[:64]
    graph=canonical_neighbors(X,metadata['config'].get('num_neighbors',16))
    times=torch.linspace(min(f['time'] for f in frames),max(f['time'] for f in frames),8,device=X.device)
    buffers={key:[] for key in ('X','F','valid','confidence','pred','base_residual','time','particle_ids','labels')}
    for time in times:
        expanded=time.expand(len(X),1)
        with torch.no_grad():
            output=deform.step(X,expanded)['d_xyz']
            x=(quaternion_to_matrix(output[:,:4])@X[...,None]).squeeze(-1)+output[:,4:]
            result=estimate_F(X,x,graph,ids,min_determinant=metadata['config'].get('min_determinant',1e-6))
        particle_ids=gaussians.particle_indices[ids]
        current=torch.cat((X[ids],time.expand(len(ids),1)),dim=-1).requires_grad_(True)
        pred=predicted_stress(current,particle_ids,gaussians.velocity_net,gaussians.force_sigma)
        legacy=torch.cat((X[ids],(time/len(frames)).expand(len(ids),1)),dim=-1).requires_grad_(True)
        inertial,divergence=momentum_components(legacy,lambda value:gaussians.velocity_net(value,particle_ids),gaussians.force_sigma)
        labels=extension.material_ids[particle_ids.cpu()] if extension.material_ids is not None else torch.zeros(len(ids),dtype=torch.long)
        values={'X':X[ids],'F':result['F'],'valid':result['valid'],'confidence':result['confidence'],
                'pred':pred,'base_residual':inertial-divergence,'time':time.expand(len(ids)),
                'particle_ids':particle_ids,'labels':labels}
        for key,value in values.items():buffers[key].append(value.detach().cpu())
    data={key:torch.cat(values).double() if key not in ('valid','particle_ids','labels') else torch.cat(values)
          for key,values in buffers.items()}
    params=extension.material.values() if hasattr(extension.material,'values') else extension.material() if extension.material is not None else {}
    mu=params['mu'].detach().reshape(-1).double() if params else torch.empty(0,dtype=torch.float64)
    kappa=params['kappa'].detach().reshape(-1).double() if params else torch.empty(0,dtype=torch.float64)
    materials=len(mu);parts=[mu.log(),kappa.log()]
    force_shape=None
    if extension.force is not None and extension.force.coefficients is not None:
        force_shape=extension.force.coefficients.shape;parts.append(extension.force.coefficients.detach().flatten().double())
    theta=torch.cat(parts)
    labels=data['labels'];valid=data['valid'];force=extension.force
    def residual(value):
        rows=[]
        if materials:
            material={'mu':value[:materials].exp()[labels[valid]],'kappa':value[materials:2*materials].exp()[labels[valid]]}
            const=NeoHookean()(data['F'][valid],material)
            rows.append(((data['pred'][valid]-const)*data['confidence'][valid,None,None].sqrt()).flatten())
        ext=force(data['X'],data['time'],data['particle_ids'],value[2*materials:].reshape(force_shape) if force_shape is not None else None) if force is not None else torch.zeros_like(data['X'])
        rows.append((data['base_residual']-ext).flatten())
        return torch.cat(rows)
    report,J=jacobian_diagnostics(residual,theta,materials*2)
    names=[f'log_mu_{i}' for i in range(materials)]+[f'log_kappa_{i}' for i in range(materials)]+[f'force_control_{i}' for i in range(len(theta)-materials*2)]
    controlled_errors=None
    gt_path=Path(dataset.source_path)/'ground_truth.npz'
    if gt_path.exists():
        import numpy as np
        gt=np.load(gt_path,allow_pickle=False)
        truth=StructuredForce({'force_mode':'known','known_force_file':str(gt_path)})
        true_force=truth(data['X'],data['time'],data['particle_ids'])
        predicted=force(data['X'],data['time'],data['particle_ids']).detach() if force is not None else torch.zeros_like(true_force)
        controlled_errors={'origin':'manufactured reference only',
                           'mu_relative_error':(abs(mu-torch.as_tensor(gt['mu']))/torch.as_tensor(gt['mu'])).tolist() if len(mu)==len(gt['mu']) else None,
                           'kappa_relative_error':(abs(kappa-torch.as_tensor(gt['kappa']))/torch.as_tensor(gt['kappa'])).tolist() if len(kappa)==len(gt['kappa']) else None,
                           'force_relative_l2_error':((predicted-true_force).norm()/true_force.norm().clamp_min(1e-12)).item()}
    result={'checkpoint_iteration':state['iteration'],'checkpoint_sha256':sha256(run/'checkpoints/resume.pt'),
            'diagnostic_snapshot':'final full checkpoint; reconstruction metrics may use a different best-test snapshot',
            'parameter_names':names,'force_shape':list(force_shape) if force_shape is not None else None,
            'conditional_physical_residual':report,'manufactured_ambiguity':ambiguity_experiment(),
            'controlled_reference_errors':controlled_errors,
            'caveats':['Neural geometry, velocity and stress fields are frozen for this Jacobian. Their uncertainty is excluded.',
                       'This conditional residual Jacobian does not prove observational or global identifiability.',
                       'Stress and CMR time conventions differ in the retained baseline; no force recovery is claimed.',
                       'Rows use dataset stress/momentum units with unit weights; conditioning depends on parameterization/scales.']}
    write_json(run/'identifiability.json',result)
    torch.save({'J':J,'theta':theta,'parameter_names':names},run/'conditional_jacobian.pt')
    print(json.dumps({'parameter_count':len(theta),'rank':report['effective_rank'],'condition_number':report['condition_number'],
                      'controlled_reference_errors':controlled_errors}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    diagnose(parser.parse_args().run)
