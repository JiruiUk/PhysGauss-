"""Fixed train-only reference patches; original C1 path when shape flags are off."""
import json
from pathlib import Path
import torch
from variants.color_repair.runtime import Tuning as Parent
from variants.metric_tuning.runtime import read_json,write_json,sha256
from .model import local_shape


class Tuning(Parent):
    def __init__(self,config_path,run):
        super().__init__(config_path,run)
        self.shape_enabled=self.enabled and self.config.get('shape_enabled',False)
        self.graph=None
        if self.shape_enabled:
            for name in ['shape_affine_weight','shape_collapse_weight']:
                if self.config.get(name,0)<0:raise ValueError('Negative shape weight')
            if self.config.get('affine_weight',0) or self.config.get('coverage_density',False):
                raise ValueError('New local module must not mix legacy affine or point insertion')

    def after_scene(self,gaussians,deform):
        super().after_scene(gaussians,deform)
        if self.shape_enabled:self.shape_deform=deform

    def configure_scene(self,scene,source):
        super().configure_scene(scene,source)
        if self.shape_enabled:
            self.shape_data=Path(source)
            self.shape_metadata=read_json(self.shape_data/'transforms_train.json')

    def coverage_loss(self,iteration,view,package,gaussians,warp):
        inherited=super().coverage_loss(iteration,view,package,gaussians,warp)
        if not self.shape_enabled:return inherited
        start=int(self.config.get('shape_start',16000))
        if iteration<start:return inherited
        if self.graph is None:
            from .observations import infer_reference
            attention=self.shape_deform.deform.attention_score
            with torch.random.fork_rng(devices=[gaussians.get_xyz.device]):
                self.graph=infer_reference(gaussians,self.shape_deform,self.shape_metadata,self.shape_data,
                                           self.run/'shape_reference')
            self.shape_deform.deform.attention_score=attention
            torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in self.graph.items()},
                       self.run/'shape_reference/reference_graph.pt')
            self.append('shape_loss.jsonl',{'iteration':iteration,'event':'graph_built',
                'points':self.graph['point_count'],'anchors':len(self.graph['anchors']),
                'known_counts':[int((self.graph['labels']==i).sum()) for i in range(4)],
                'only_training_reference_frames':self.graph['frame_ids']})
        if len(gaussians.get_xyz)!=self.graph['point_count']:raise ValueError('Frozen graph kernel topology changed')
        interval=int(self.config.get('shape_interval',20))
        if iteration<=start or iteration%interval:return inherited
        if not torch.is_tensor(warp):raise ValueError('Complete deformation warp required')
        from variants.kinematic_tuning.model import center_trajectory
        g=self.graph;a=g['anchors'];limit=min(len(a),int(self.config.get('shape_samples',128)))
        ids=(torch.arange(limit,device=a.device)+iteration//interval*limit)%len(a)
        r=local_shape(g['reference'],center_trajectory(gaussians.get_xyz.detach(),warp),a[ids],g['neighbors'][ids],g['valid'][ids],
                      self.config.get('shape_min_stretch',.3),self.config.get('shape_max_stretch',3.))
        ramp=min(1.,(iteration-start)/max(1,int(self.config.get('shape_ramp',2000))))
        loss=interval*ramp*(self.config.get('shape_affine_weight',0)*r['affine']+
                            self.config.get('shape_collapse_weight',0)*r['collapse'])
        if not torch.isfinite(loss):raise ValueError('Nonfinite local shape objective')
        if iteration%100==0 or self.config.get('shape_audit',False):
            self.append('shape_loss.jsonl',{'iteration':iteration,'frame':view.image_name,'affine':float(r['affine'].detach()),
                'collapse':float(r['collapse'].detach()),'weighted_loss':float(loss.detach()),'ramp':ramp,
                'min_stretch_p05':float(r['singular'][:,-1].detach().quantile(.05)),
                'max_stretch_p95':float(r['singular'][:,0].detach().quantile(.95))})
        return loss if inherited is None else inherited+loss

    def save(self,checkpoint,manifest):
        super().save(checkpoint,manifest)
        if not self.shape_enabled:return
        p=Path(checkpoint).with_suffix('.shape.pt');tmp=p.with_suffix('.tmp')
        graph=None if self.graph is None else {k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in self.graph.items()}
        torch.save({'iteration':manifest['iteration'],'checkpoint_sha256':manifest['sha256'],'graph':graph},tmp);tmp.replace(p)
        sidecar=Path(checkpoint).with_suffix('.metrics.json');s=read_json(sidecar)
        s['shape_graph_sha256']=sha256(p);write_json(sidecar,s)

    def restore(self,checkpoint,iteration):
        super().restore(checkpoint,iteration)
        if not self.shape_enabled:return
        p=Path(checkpoint).with_suffix('.shape.pt')
        if not p.exists() and iteration<int(self.config.get('shape_start',16000)):return
        sidecar=read_json(Path(checkpoint).with_suffix('.metrics.json'))
        if not p.exists() or sha256(p)!=sidecar.get('shape_graph_sha256'):raise ValueError('Wrong shape graph SHA')
        s=torch.load(p,map_location='cpu',weights_only=True)
        if s['iteration']!=iteration or s['checkpoint_sha256']!=sha256(checkpoint):raise ValueError('Wrong shape graph boundary')
        self.graph=None if s['graph'] is None else {k:v.cuda() if torch.is_tensor(v) else v for k,v in s['graph'].items()}
