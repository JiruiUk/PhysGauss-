"""Read-only final-snapshot reproduction and velocity-parameter gradient probes."""
from pathlib import Path
import os,sys,json,runpy,hashlib
ROOT=Path(r'F:/A_Edinburgh/PhysGauss');RUN=ROOT/'runs/pidg_monotonic_training/J1_monotonic_kinematic/rubber_duck';QA=ROOT/'output/monotonic_training/verification_20261102'
sys.path.insert(0,str(ROOT/'variants/monotonic_training/code'));sys.path.insert(1,str(ROOT))
os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder');os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
import torch,numpy as np
from plyfile import PlyData
from motion_utils.pidg import VelocityNet,cauchy_momentum_constraint
from scene_PIDG.deform_model import DeformModel
from variants.monotonic_training.model import configure_clock,time_coordinate
from variants.kinematic_tuning.model import center_trajectory,trajectory_velocity
from variants.frozen_velocity_probe.core import state_hash,alignment_metrics
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
torch.set_num_threads(4)
p=RUN/'point_cloud/iteration_50000';material=torch.load(p/'material.pt',map_location='cuda',weights_only=True)
net=VelocityNet().cuda();net.load_state_dict(material['velocity_net']);configure_clock(net,'fixed_monotonic');net.eval();net_before=state_hash(net)
ply=PlyData.read(p/'point_cloud.ply')['vertex'];X=torch.tensor(np.column_stack([ply[k] for k in ['x','y','z']]),device='cuda');N=len(X)
ids=torch.arange(256,device='cuda')*N//256;x=X[ids].detach();pid=material['particle_indices'][ids]
cfg=runpy.run_path(str(RUN/'model_config.py'));deform=DeformModel(cfg['grid_args'],cfg['network_args'],reg_spatial_able=False,reg_temporal_able=False);deform.load_weights(str(RUN),50000)
for module in [deform.grid4d,deform.deform]:module.eval().requires_grad_(False)
teacher_before=[state_hash(m) for m in [deform.grid4d,deform.deform]]
full=torch.load(RUN/'checkpoints/resume.pt',map_location='cpu',weights_only=False)
assert torch.equal(full['gaussian_parameters']['_xyz'],X.cpu())
assert all(torch.equal(v.cpu(),full['velocity_net'][k]) for k,v in net.state_dict().items())
assert all(torch.equal(v.cpu(),full['grid4d'][k]) for k,v in deform.grid4d.state_dict().items())
assert all(torch.equal(v.cpu(),full['deform'][k]) for k,v in deform.deform.state_dict().items())
prefix=torch.load(ROOT/'runs/pidg_color_repair/shared_prefix/rubber_duck/checkpoints/resume.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(v.cpu(),prefix['velocity_net'][k]) for k,v in net.state_dict().items() if k.startswith(('time_embed.','time_proj.')))
del full,prefix

def trajectory(x,t):
 deform.deform.attention_score=None
 return center_trajectory(x,deform.step(x,t,fixed_attention=False)['d_xyz'])
log=read(RUN/'clock_contract.json');assert log['velocity_clock_mode']=='fixed_monotonic'
tgrid=torch.linspace(0,1,1001,device='cuda')[:,None]
assert (time_coordinate(net,tgrid).diff()>0).all()
records=[json.loads(s) for s in (RUN/'kinematic_diagnostics.jsonl').read_text().splitlines()][-1];repro=[];gradient_records=[]
parameters=[(name,p) for name,p in net.named_parameters() if p.requires_grad]
for j,row in enumerate(records['times']):
 t=x.new_full((len(x),1),row['time'])
 with torch.no_grad():
  target=trajectory_velocity(trajectory,x,t,.001);pred=net(torch.cat([x,t],-1),pid)[0]
  metric=alignment_metrics(pred,target,max(target.square().mean().sqrt().item()*.1,1e-8))
  discrepancy=abs(metric['relative_rmse']-row['relative_rmse']);assert discrepancy<2e-4
 repro.append({'time':row['time'],'metrics':metric,'stored_relative_rmse':row['relative_rmse'],'absolute_difference':discrepancy})
 if j not in [1,2,3]:continue
 coords=torch.cat([x,t],-1).detach().requires_grad_(True)
 v,_=net(coords,pid);kin=torch.nn.functional.huber_loss(v,target,delta=.1)
 kg=torch.autograd.grad(kin,[p for _,p in parameters],allow_unused=True)
 coords2=coords.detach().requires_grad_(True)
 cmr=cauchy_momentum_constraint(coords2,lambda c:net(c,pid),material['force_sigma'])
 cg=torch.autograd.grad(cmr,[p for _,p in parameters],allow_unused=True)
 groups={}
 # Same-update coefficient at a common /100 step. Time-average divides by event intervals.
 kw=.001*10;cw=.01*min(N,20000)/N
 for group in ['planes','index_embed','shared_net','velocity_head','stress_head']:
  indices=[i for i,(name,_) in enumerate(parameters) if name.split('.')[0]==group]
  ka=torch.cat([(kg[i] if kg[i] is not None else torch.zeros_like(parameters[i][1])).detach().flatten() for i in indices])
  ca=torch.cat([(cg[i] if cg[i] is not None else torch.zeros_like(parameters[i][1])).detach().flatten() for i in indices])
  assert torch.isfinite(ka).all() and torch.isfinite(ca).all()
  kn,cn=ka.norm().item(),ca.norm().item()
  cosine=torch.dot(ka,ca).item()/(kn*cn) if kn*cn>0 else None
  groups[group]={'kin_raw_grad_norm':kn,'cmr_raw_grad_norm':cn,'cosine':cosine,
    'same_update_cmr_over_kin_weighted_norm':cw*cn/(kw*kn) if kn>0 else None,
    'frequency_averaged_cmr_over_kin_norm':(cw/100)*cn/((kw/10)*kn) if kn>0 else None}
 gradient_records.append({'time':row['time'],'kinematic_huber':kin.item(),'cmr_raw':cmr.item(),'groups':groups})
 print('REPRO',row['time'],metric['relative_rmse'],'head_gradient',groups['velocity_head'],flush=True)
assert state_hash(net)==net_before
assert teacher_before==[state_hash(m) for m in [deform.grid4d,deform.deform]]
summary={'date':'2026-11-02','snapshot_iteration':50000,'snapshot_matches_full_checkpoint':True,
 'unused_legacy_clock_weights_match_shared_prefix':True,'clock_strictly_monotonic':True,'no_optimizer_created':True,
 'model_parameters_unchanged':True,'diagnostic_reproduction':repro,'gradient_probes':gradient_records,
 'gradient_probe_limit':'256 deterministic points at3 training times, same points for each loss; not actual20k CMR batches or a proof of whole-training interference.',
 'supervision_budget':{'joint_velocity_optimizer_updates':4699,'joint_samples_per_update':256,'joint_sample_pairs':4699*256,
  'prior_frozen_probe_updates':12000,'prior_frozen_probe_batch':4096,'prior_frozen_probe_sample_pairs':12000*4096,
  'prior_frozen_to_joint_sample_pair_ratio':12000*4096/(4699*256),
  'other_differences':['frozen versus evolving targets','normalized MSE versus weighted Huber','CMR and image/neighbor objectives present only in joint training']}}
for dest in [QA/'physics_verification.json',ROOT/'experiments/monotonic_training/physics_verification_20261102.json']:
 dest.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
print('PHYSICS VERIFICATION COMPLETE',flush=True)
