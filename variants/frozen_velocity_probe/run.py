"""Reproducible, isolated GPU diagnostic with read-only reconstruction sources."""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

from .core import disabled

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'experiments/frozen_velocity_probe'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def initialize():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
    os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
    sys.path.insert(0,str(ROOT/'variants/integrity_tuning/code'))
    import numpy as np
    import torch
    from .core import freeze,state_hash,velocity_parameters,normalized_mse,alignment_metrics
    from motion_utils.pidg import VelocityNet
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    return np,torch,VelocityNet


def export(config,out):
    np,torch,VelocityNet=initialize()
    from plyfile import PlyData
    from scene_PIDG.deform_model import DeformModel
    from variants.kinematic_tuning.model import center_trajectory,trajectory_velocity
    from .core import freeze,state_hash,alignment_metrics,partition_candidates
    source=ROOT/config['source_run'];ntrain=config['train_particles'];nhold=config['heldout_particles']
    paths=[source/'point_cloud/iteration_50000/point_cloud.ply',source/'point_cloud/iteration_50000/material.pt',
           source/'deform/iteration_50000/deform.pth',source/'model_config.py',source/'metric_config.json']
    data=ROOT/'data/derived/kinematic_resplit/training'
    paths += [data/'transforms_train.json',data/'transforms_test.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for file in (ROOT/'variants/integrity_tuning/code').rglob('*.py'):
        if 'tmp_build' not in str(file):hashes[str(file.relative_to(ROOT))]=sha(file)
    for file in [ROOT/'variants/kinematic_tuning/model.py',Path(__file__),Path(__file__).with_name('core.py'),REPORT/'config.json',REPORT/'protocol.md']:
        hashes[str(file.relative_to(ROOT))]=sha(file)
    ply=PlyData.read(paths[0])['vertex']
    X=torch.tensor(np.column_stack([ply[k] for k in ('x','y','z')]),device='cuda')
    opacity=torch.sigmoid(torch.tensor(np.asarray(ply['opacity']).copy(),device='cuda'))
    candidates=torch.where(opacity>=config['minimum_opacity'])[0]
    material=torch.load(paths[1],map_location='cuda',weights_only=True)
    chosen=partition_candidates(candidates,material['particle_indices'],ntrain,nhold,config['seed'])
    assert len(chosen)==ntrain+nhold and len(chosen.unique())==len(chosen)
    net=VelocityNet().cuda();net.load_state_dict(material['velocity_net']);freeze(net)
    model=runpy.run_path(str(source/'model_config.py'))
    deform=DeformModel(model['grid_args'],model['network_args'],reg_spatial_able=False,reg_temporal_able=False)
    deform.load_weights(str(source),50000);freeze(deform.grid4d);freeze(deform.deform)
    teacher_hash={k:state_hash(m) for k,m in [('grid',deform.grid4d),('deformation',deform.deform)]}
    def trajectory(x,t):
        deform.deform.attention_score=None
        return center_trajectory(x,deform.step(x,t,fixed_attention=False)['d_xyz'])
    def times(path):
        frames=sorted(read(path)['frames'],key=lambda f:int(Path(f['file_path']).stem))
        ids=[int(Path(f['file_path']).stem) for f in frames]
        return ids,torch.tensor([(i-1)/199 for i in ids],device='cuda')
    train_ids,train_times=times(paths[5]);val_ids,val_times=times(paths[6])
    assert len(train_ids)==120 and len(val_ids)==30 and not set(train_ids)&set(val_ids)
    targets={};centers={};xs=X[chosen].detach()
    start=time.monotonic()
    with torch.no_grad():
        for split,ts in [('train',train_times),('validation',val_times)]:
            vals=[];cs=[]
            for i,tv in enumerate(ts):
                t=xs.new_full((len(xs),1),tv.item())
                c=trajectory(xs,t);v=trajectory_velocity(trajectory,xs,t,config['derivative_h'])
                assert torch.isfinite(c).all() and torch.isfinite(v).all()
                vals.append(v.cpu());cs.append(c.cpu())
                if i%30==0:print('EXPORT',split,i,'elapsed',round(time.monotonic()-start,1),flush=True)
            targets[split]=torch.stack(vals);centers[split]=torch.stack(cs)
        # Exact old diagnostic subset and timestamps, separate from fitting split.
        old=[json.loads(s) for s in (source/'kinematic_diagnostics.jsonl').read_text().splitlines() if s.strip()][-1]
        legacy_idx=torch.arange(256,device='cuda')*len(X)//256
        legacy=[];sens=[]
        legacy_targets=[];legacy_coords=[];legacy_pids=[]
        for row in old['times']:
            tv=row['time'];lx=X[legacy_idx];lt=lx.new_full((len(lx),1),tv)
            v=trajectory_velocity(trajectory,lx,lt,config['derivative_h'])
            pred=net(torch.cat([lx,lt],-1),material['particle_indices'][legacy_idx])[0]
            measured=alignment_metrics(pred,v,0.)
            diff=abs(measured['relative_rmse']-row['relative_rmse'])
            assert diff<2e-4,(tv,diff)
            legacy.append({'time':tv,'stored_relative_rmse':row['relative_rmse'],'recomputed':measured,'absolute_difference':diff})
            legacy_targets.append(v.cpu());legacy_coords.append(torch.cat([lx,lt],-1).cpu());legacy_pids.append(material['particle_indices'][legacy_idx].cpu())
            for h in (.00025,.0005,.002):
                alt=trajectory_velocity(trajectory,lx,lt,h)
                sens.append({'time':tv,'h':h,**alignment_metrics(alt,v,max(v.square().mean().sqrt().item()*.1,1e-8))})
        # h vs h/2 is a sensitivity measure, not proof of a true physical derivative.
        t=xs.new_full((len(xs),1),train_times[len(train_times)//2].item())
        first=trajectory_velocity(trajectory,xs,t,config['derivative_h'])
        repeated=trajectory_velocity(trajectory,xs,t,config['derivative_h'])
        assert torch.equal(first,repeated)
    assert all(p.grad is None for m in [deform.grid4d,deform.deform] for p in m.parameters())
    assert teacher_hash=={k:state_hash(m) for k,m in [('grid',deform.grid4d),('deformation',deform.deform)]}
    payload={'X':xs.cpu(),'particle_indices':material['particle_indices'][chosen].cpu(),'gaussian_rows':chosen.cpu(),
             'train_times':train_times.cpu(),'validation_times':val_times.cpu(),'targets':targets,'centers':centers,
             'ntrain':ntrain,'legacy':{'coords':torch.cat(legacy_coords),'ids':torch.cat(legacy_pids),'target':torch.cat(legacy_targets)},
             'source_velocity':{k:v.cpu() for k,v in net.state_dict().items()}}
    torch.save(payload,out/'targets.pt')
    write(out/'manifest.json',dict(config=config,source_sha256=hashes,targets_sha256=sha(out/'targets.pt'),
          train_frame_ids=train_ids,validation_frame_ids=val_ids,total_gaussians=len(X),eligible_opacity_points=len(candidates),
          teacher_state_hash=teacher_hash,teacher_gradients_absent=True,repeated_target_bitwise_equal=True,
          heldout_particle_id_overlap=0,unique_train_particle_ids=len(material['particle_indices'][chosen[:ntrain]].unique()),
          unique_heldout_particle_ids=len(material['particle_indices'][chosen[ntrain:]].unique()),
          original_diagnostic_reproduction=legacy,derivative_sensitivity=sens,
          torch_version=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
          export_seconds=time.monotonic()-start,units='scene distance / normalized time; not m/s',
          final_test_opened=False))
    print('EXPORT COMPLETE',sha(out/'targets.pt'),flush=True)


def datasets(data,torch):
    n=data['ntrain'];X=data['X'].cuda();ids=data['particle_indices'].cuda()
    result={}
    for split in ['train','validation']:
        ts=data[split+'_times'].cuda()
        for part,selection in [('known',slice(0,n)),('heldout',slice(n,None))]:
            x=X[selection];pid=ids[selection]
            coords=torch.cat([x[None].expand(len(ts),-1,-1),ts[:,None,None].expand(-1,len(x),1)],-1).reshape(-1,4)
            result[part+'_'+split]={'coords':coords,'ids':pid.repeat(len(ts)),
              'target':data['targets'][split][:,selection].cuda().reshape(-1,3)}
    result['legacy']={k:v.cuda() for k,v in data['legacy'].items()}
    return result


def train(config,out,smoke=False):
    np,torch,VelocityNet=initialize()
    from .core import state_hash,velocity_parameters,normalized_mse,alignment_metrics
    cache=out/'targets.pt';manifest=read(out/'manifest.json');assert sha(cache)==manifest['targets_sha256']
    data=torch.load(cache,map_location='cpu',weights_only=True);sets=datasets(data,torch)
    assert not torch.isin(data['particle_indices'][:data['ntrain']],data['particle_indices'][data['ntrain']:]).any()
    scale=sets['known_train']['target'].square().mean().sqrt().clamp_min(1e-8)
    threshold=scale.item()*.1
    # A uses a fixed subset of the exact same cached labels as the broad fit.
    ti=torch.linspace(0,len(data['train_times'])-1,config['overfit_times']).long()
    ai=(ti[:,None]*data['ntrain']+torch.arange(config['overfit_particles'])[None]).flatten().cuda()
    small={k:v[ai] for k,v in sets['known_train'].items()}
    arms=[dict(name='A_small_overfit',initialization='pretrained',lr=.001,seed=config['seed'],steps=config['overfit_steps'])]
    arms += [dict(a,steps=config['general_steps']) for a in config['arms']]
    if smoke:arms=[dict(arms[0],name='_smoke',steps=100)]
    all_results=[]
    for arm in arms:
        torch.manual_seed(arm['seed']);torch.cuda.manual_seed_all(arm['seed'])
        net=VelocityNet().cuda()
        if arm['initialization']=='pretrained':net.load_state_dict(data['source_velocity'])
        params=velocity_parameters(net);net.train();stress_hash=state_hash(net.stress_head)
        held_ids=data['particle_indices'][data['ntrain']:].unique().cuda()
        held_embedding_before=net.index_embed.weight[held_ids].detach().clone()
        fit=small if arm['name'].startswith('A_') or smoke else sets['known_train']
        fit_scale=fit['target'].square().mean().sqrt().clamp_min(1e-8)
        optimizer=torch.optim.Adam(params,lr=arm['lr'])
        gen=torch.Generator(device='cuda').manual_seed(arm['seed'])
        arm_dir=out/arm['name'];arm_dir.mkdir(exist_ok=True)
        log=arm_dir/'history.jsonl'
        if log.exists():raise ValueError('Refusing to overwrite existing experiment '+str(log))
        start=time.monotonic();gradients={}
        @torch.no_grad()
        def evaluate(selection):
            pred=[]
            for j in range(0,len(selection['coords']),8192):
                pred.append(net(selection['coords'][j:j+8192],selection['ids'][j:j+8192])[0])
            return alignment_metrics(torch.cat(pred),selection['target'],threshold)
        initial={k:evaluate(v) for k,v in dict(fit=fit,**sets).items()}
        def record(step,loss=None):
            row=dict(step=step,seconds=time.monotonic()-start,loss=loss,metrics={k:evaluate(v) for k,v in dict(fit=fit,**sets).items()})
            with log.open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            print(arm['name'],'step',step,'fit',round(row['metrics']['fit']['relative_rmse'],5),
                  'time-val',round(row['metrics']['known_validation']['relative_rmse'],5),
                  'cross',round(row['metrics']['heldout_validation']['relative_rmse'],5),
                  'sec',round(row['seconds'],1),flush=True)
            return row
        record(0)
        for step in range(1,arm['steps']+1):
            if len(fit['coords'])<=config['batch_size']:indices=torch.arange(len(fit['coords']),device='cuda')
            else:indices=torch.randint(len(fit['coords']),(config['batch_size'],),generator=gen,device='cuda')
            lr=arm['lr']*(.1+.9*(1+math.cos(math.pi*(step-1)/arm['steps']))/2)
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            prediction=net(fit['coords'][indices],fit['ids'][indices])[0]
            loss=normalized_mse(prediction,fit['target'][indices],fit_scale)
            assert torch.isfinite(loss)
            loss.backward()
            if step==1 or step%1000==0:
                for name,p in net.named_parameters():
                    if p.requires_grad:
                        assert p.grad is not None and torch.isfinite(p.grad).all(),name
                        if step==1:gradients[name]=p.grad.norm().item()
                assert net.stress_head.weight.grad is None
            optimizer.step()
            if step%config['evaluation_interval']==0 or step==arm['steps']:final=record(step,loss.item())
        assert state_hash(net.stress_head)==stress_hash
        assert torch.equal(held_embedding_before,net.index_embed.weight[held_ids])
        torch.save({'velocity_net':{k:v.cpu() for k,v in net.state_dict().items()},'optimizer':optimizer.state_dict(),
                    'arm':arm,'target_sha256':manifest['targets_sha256'],'diagnostic_only':True},arm_dir/'velocity_only.pt')
        result=dict(arm=arm,initial=initial,final=final,gradient_norms_first_step=gradients,
                    unused_stress_head_parameters_unchanged=True,source_renderer_never_optimized=True,
                    heldout_id_embedding_bitwise_unchanged=True,
                    trained_parameter_count=sum(p.numel() for p in params),seconds=time.monotonic()-start)
        write(arm_dir/'result.json',result);all_results.append(result)
        del net,optimizer;gc.collect();torch.cuda.empty_cache()
    # Ensure every inherited Python source and every reconstruction input is unchanged.
    checks={p:sha(ROOT/p)==value for p,value in manifest['source_sha256'].items()}
    assert all(checks.values()),checks
    assert sha(cache)==manifest['targets_sha256']
    result=dict(source=config['source_run'],target_scale=scale.item(),movement_threshold=threshold,
                arms=all_results,source_hashes_unchanged=checks,targets_unchanged=True,final_test_opened=False)
    write(out/('smoke_result.json' if smoke else 'results.json'),result)
    if not smoke:write(REPORT/'results.json',result)
    print('ALL COMPLETE',out,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=REPORT/'config.json')
    parser.add_argument('--phase',choices=['export','smoke','train','all'],default='all');args=parser.parse_args()
    config=read(args.config)
    if disabled(config):print('disabled: no model imports, checkpoint access or optimization');return
    out=ROOT/config['output_run'];out.mkdir(parents=True,exist_ok=True)
    if args.phase in ['export','all']:export(config,out)
    if args.phase in ['smoke','train','all']:train(config,out,smoke=args.phase=='smoke')


if __name__=='__main__':main()
