"""Training hooks; disabled mode creates no parameters and consumes no RNG."""
import json
from pathlib import Path
import torch
from .deformation_gradient import CanonicalGraph, estimate_F


class Extension:
    def __init__(self,config_path,run):
        self.config=json.loads(Path(config_path).read_text()) if config_path else {}
        self.enabled=self.config.get('deformation_gradient',False)
        self.run=Path(run)
        if self.enabled:
            self.graph=CanonicalGraph(self.config.get('num_neighbors',16))
        self.material=None
        if self.config.get('constitutive_diagnostics') or self.config.get('constitutive'):
            from .constitutive import MaterialParameters,NeoHookean
            self.material=MaterialParameters(self.config.get('mu_init',1.),self.config.get('kappa_init',10.),
                                             self.config.get('learn_material_parameters',False))
            self.constitutive=NeoHookean()

    def loss(self,iteration,gaussians,deform,d_xyz,fid,opt):
        if not self.enabled or iteration % self.config.get('interval',100):
            return None
        from utils.general_utils import quaternion_to_matrix
        with torch.no_grad():
            X=gaussians.get_xyz
            x=X if not torch.is_tensor(d_xyz) else (quaternion_to_matrix(d_xyz[...,:4])@X[...,None]).squeeze(-1)+d_xyz[...,4:]
            graph=self.graph.get(X,iteration)
            ids=torch.arange(0,len(X),max(1,len(X)//256),device=X.device)[:256]
            result=estimate_F(X,x,graph,ids)
            record={'iteration':iteration,'mean_det_F':result['J'].mean().item(),
                    'min_det_F':result['J'].min().item(),'invalid_F_fraction':1-result['valid'].float().mean().item(),
                    'diagnostic_queries':len(ids),'new_loss_added':False}
            if self.material is not None:
                valid=result['valid']
                params=self.material()
                sigma=self.constitutive(result['F'][valid],params)
                record.update(mean_sigma_const_norm=sigma.norm(dim=(-1,-2)).mean().item() if valid.any() else None,
                              mu=params['mu'].item(),kappa=params['kappa'].item())
        with (self.run/'extension.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        return None
