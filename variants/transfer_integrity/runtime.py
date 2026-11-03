"""Independent opt-in loss branch; exact repaired-C1 delegation when disabled."""
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import torch
from variants.color_repair.runtime import Tuning as Parent
from variants.metric_tuning.runtime import read_json,write_json,sha256
from variants.integrity_tuning.model import coverage_objective
from variants.flow_comparison.temporal import deterministic
from variants.kinematic_tuning.model import center_trajectory
from variants.flow_comparison.flow import rotation_raw
from output.short_clip_diagnostic.camera_math import w2c,intrinsic
from .model import static_position,static_covariance,static_anchor,reprojection,local_shape


class Tuning(Parent):
    def __init__(self,config_path,run):
        super().__init__(config_path,run)
        self.transfer_enabled=self.enabled and self.config.get('transfer_integrity_enabled',False)
        if self.transfer_enabled:
            for name in ['observed_coverage_weight','static_position_weight','static_covariance_weight','static_anchor_weight','track_reprojection_weight','local_shape_weight']:
                value=float(self.config.get(name,0))
                if value<0 or not math.isfinite(value):raise ValueError('Invalid transfer loss weight: '+name)

    def after_scene(self,g,d):
        super().after_scene(g,d)
        if self.transfer_enabled:self.transfer_deform=d

    def initialize_data(self,source):
        path=Path(self.config['transfer_data'])
        if sha256(path)!=self.config['transfer_data_sha256']:raise ValueError('Transfer binding changed')
        with np.load(path) as z:self.binding={k:z[k].copy() for k in z.files}
        if str(self.binding['source_checkpoint_sha256'])!=self.config['transfer_source_sha256']:raise ValueError('Wrong source state')
        meta=read_json(Path(source)/'transforms_train.json')
        self.transfer_frames={int(Path(f['file_path']).stem):f for f in meta['frames']}
        self.transfer_numbers=sorted(self.transfer_frames)
        self.transfer_cameras={}
        k=intrinsic(meta['camera_angle_x'],'legacy')
        for n,f in self.transfer_frames.items():
            self.transfer_cameras[n]=(torch.tensor(w2c(f['transform_matrix'],'legacy'),device='cuda',dtype=torch.float32),
                                      torch.tensor(k,device='cuda',dtype=torch.float32))

    def configure_scene(self,scene,source):
        super().configure_scene(scene,source)
        if self.transfer_enabled:self.initialize_data(source)

    def at(self,g,ids,time,detach_canonical=False):
        x=g.get_xyz[ids]
        if detach_canonical:x=x.detach()
        t=x.new_full((len(x),1),time) if isinstance(time,(int,float)) else torch.as_tensor(time,device=x.device,dtype=x.dtype).reshape(-1,1)
        d=self.transfer_deform;old=d.deform.attention_score
        try:
            with deterministic(d):w=d.step(x,t,fixed_attention=False)
        finally:d.deform.attention_score=old
        centers=center_trajectory(x,w['d_xyz'])
        rot=rotation_raw(g.get_rotation[ids]+w['d_rotation'])
        scale=g.get_scaling[ids]+w['d_scaling']
        covariance=(rot*scale[:,None,:].square())@rot.transpose(-1,-2)
        return centers,covariance

    def geometry_loss(self,iteration,g):
        b=self.binding
        if len(g.get_xyz)!=int(b['point_count']):raise ValueError('Bound Gaussian topology changed')
        interval=int(self.config.get('transfer_interval',10));index=(iteration-50000)//interval-1
        limit=int(self.config.get('static_samples',1024))
        choice=(np.arange(min(limit,len(b['static_ids'])))+index*limit)%len(b['static_ids'])
        ids=torch.as_tensor(b['static_ids'][choice],device='cuda',dtype=torch.long)
        n=self.transfer_numbers[index*37%len(self.transfer_numbers)]
        now,cov=self.at(g,ids,(n-1)/199,True)
        ref,refcov=self.at(g,ids,(b['static_reference_frames'][choice]-1)/199,True)
        terms={'static_position':static_position(now,ref),'static_covariance':static_covariance(cov,refcov)}
        anchors=torch.tensor(b['anchor_ids'],device='cuda',dtype=torch.long)
        anchored,_=self.at(g,anchors,(n-1)/199)
        target=torch.tensor(b['anchor_xyz'],device='cuda')
        terms['static_anchor']=static_anchor(anchored,target)
        batch=index*11%int(b['batch_count']);prefix=f'b{batch}_'
        rows=b[prefix+'ids'];targets=b[prefix+'target'];edges=b[prefix+'edges']
        take=(np.arange(min(int(self.config.get('track_samples',512)),len(rows)))+index*512)%len(rows)
        rows=rows[take];targets=targets[take]
        if len(edges)>512:edges=edges[(np.arange(512)+index*512)%len(edges)]
        all_ids=np.unique(np.r_[rows,edges[:,:2].astype(int).ravel()])
        all_ids_t=torch.tensor(all_ids,device='cuda',dtype=torch.long)
        frame=int(b[prefix+'frame']);positions,_=self.at(g,all_ids_t,(frame-1)/199)
        lookup=torch.tensor(np.searchsorted(all_ids,rows),device='cuda')
        view,k=self.transfer_cameras[frame]
        terms['track_reprojection'],epe=reprojection(positions[lookup],view,k,torch.tensor(targets,device='cuda',dtype=torch.float32))
        if len(edges):
            a=torch.tensor(np.searchsorted(all_ids,edges[:,0].astype(int)),device='cuda')
            c=torch.tensor(np.searchsorted(all_ids,edges[:,1].astype(int)),device='cuda')
            terms['local_shape']=local_shape(positions[a],positions[c],torch.tensor(edges[:,2],device='cuda',dtype=torch.float32))
        else:terms['local_shape']=positions.sum()*0
        terms['static_displacement_RMS']=(now-ref).square().mean().sqrt().detach()
        terms['static_anchor_distance']=(anchored-target).norm(dim=-1).mean().detach()
        terms['track_EPE_400']=epe.detach()
        terms['static_sample_time_frame']=n;terms['track_frame']=frame
        loss=sum(float(self.config.get(key+'_weight',0))*terms[key] for key in ['static_position','static_covariance','static_anchor','track_reprojection','local_shape'])
        return loss,terms

    def coverage_loss(self,iteration,view,package,g,warp):
        inherited=super().coverage_loss(iteration,view,package,g,warp)
        if self.config.get('record_transfer_sampling',False):
            self.append('camera_sampling.jsonl',{'iteration':iteration,'frame':view.image_name,'time':float(view.fid)})
        if not self.transfer_enabled:return inherited
        if view.image_name not in self.train_names:raise ValueError('Transfer supervision must be train-only')
        ramp=min(1.,max(0.,(iteration-50000)/int(self.config.get('transfer_ramp',400))))
        observed=coverage_objective(package['alpha'],view.coverage_target.to(package['alpha']))
        added=float(self.config.get('observed_coverage_weight',0))*ramp*observed
        interval=int(self.config.get('transfer_interval',10))
        if iteration>50000 and (iteration-50000)%interval==0:
            geometry,terms=self.geometry_loss(iteration,g)
            added=added+interval*ramp*geometry
            record={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in terms.items()}
            record.update(iteration=iteration,observed_coverage=float(observed.detach()),weighted_loss=float(added.detach()),ramp=ramp)
            self.append('transfer_loss.jsonl',record)
        if not torch.isfinite(added):raise ValueError('Nonfinite transfer loss')
        return added if inherited is None else inherited+added

    def save(self,checkpoint,manifest):
        super().save(checkpoint,manifest)
        if self.transfer_enabled:
            path=Path(checkpoint).with_suffix('.metrics.json');s=read_json(path)
            s['transfer_data_sha256']=self.config['transfer_data_sha256'];write_json(path,s)

    def restore(self,checkpoint,iteration):
        super().restore(checkpoint,iteration)
        if self.transfer_enabled:
            s=read_json(Path(checkpoint).with_suffix('.metrics.json'))
            if s.get('transfer_data_sha256')!=self.config['transfer_data_sha256']:raise ValueError('Restored transfer binding differs')
