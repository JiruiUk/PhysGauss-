"""Freeze observation targets and partial surface bindings from full C1 state."""
from pathlib import Path
import sys
import os
import json
import numpy as np
from scipy import ndimage
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'variants/association_training/code'))
os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
from variants.metric_tuning.runtime import read_json,write_json,sha256
from variants.association_training.evaluate import load,render_call
from variants.shape_coherence.observations import camera
from variants.flow_comparison.temporal import warp_at
from variants.kinematic_tuning.model import center_trajectory
from variants.flow_comparison.flow import projected
from variants.flow_comparison.geometry import sample,forward_backward_valid
from variants.surface_correspondence.model import qualify_fragments
from output.surface_support.core import propagate
from variants.transfer_integrity.static_audit import images,inside
from variants.transfer_integrity.model import fixed_edges

SOURCE=ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck'
TRAIN=ROOT/'data/derived/kinematic_resplit/training'
REPORT=ROOT/'experiments/transfer_integrity'
RUNS=ROOT/'runs/pidg_transfer_integrity'
FLOW=ROOT/'data/derived/flow_comparison_train_only'


def extract(g,d,cam,number,masks):
    with torch.no_grad():w=warp_at(d,g.get_xyz.detach(),(number-1)/199)
    parts=[];errors=[]
    for offset in range(0,len(masks),3):
        color=torch.ones(len(g.get_xyz),3,device='cuda',requires_grad=True)
        m=torch.tensor(masks[offset:offset+3],device='cuda',dtype=torch.float32)
        p=render_call(cam,g,w,color)
        grad=torch.autograd.grad((p['render']*m).sum(),color)[0].detach()
        expected=(p['alpha']*m).sum((1,2));error=float(((grad.sum(0)-expected).abs()/expected.clamp_min(1)).max())
        assert error<5e-4 and grad.isfinite().all() and grad.min()>-1e-7
        parts.append(grad.cpu().numpy()/16);errors.append(error)
    with torch.no_grad():
        x=center_trajectory(g.get_xyz.detach(),w['d_xyz'])
    return np.concatenate(parts,axis=1),w,x,max(errors)


def prepare():
    binding_path=RUNS/'training_data.npz'
    if binding_path.exists():
        result=read_json(REPORT/'data_binding.json')
        assert sha256(binding_path)==result['sha256']
        return result
    assert read_json(REPORT/'static_image_audit.json')['static_hypothesis_gate_passed']
    meta=read_json(TRAIN/'transforms_train.json')
    frames={int(Path(f['file_path']).stem):f for f in meta['frames']}
    numbers=sorted(frames)
    clips=[c for c in read_json(ROOT/'experiments/short_clip_audit/results.json')['selected'] if c['name']!='clip4_0002_0008']
    clips.sort(key=lambda c:c['frames'][0])
    clip_frames={n for c in clips for n in c['frames']}
    tracks=np.load(ROOT/'output/surface_support/image_tracks.npz')
    components={int(n):v.copy() for n,v in zip(tracks['frames'],tracks['components'])}
    flow_rows=read_json(FLOW/'manifest.json')['pairs']
    flow={}
    files=[Path(__file__),SOURCE/'checkpoints/resume.pt',SOURCE/'metric_config.json',SOURCE/'model_config.py',
           TRAIN/'transforms_train.json',REPORT/'plan.json',REPORT/'static_image_audit.json',
           ROOT/'experiments/short_clip_audit/results.json',ROOT/'output/surface_support/image_tracks.npz',FLOW/'manifest.json']
    for row in flow_rows:
        if any(row['source'] in c['frames'] and row['target'] in c['frames'] for c in clips):
            p=FLOW/row['file'];files.append(p)
            with np.load(p) as z:flow[(row['source'],row['target'])]=(torch.tensor(z['forward']),torch.tensor(z['backward']))
    files += [TRAIN/(f['file_path']+'.png') for f in frames.values()]
    locked={str(p.relative_to(ROOT)):sha256(p) for p in files}
    assert locked[str((SOURCE/'checkpoints/resume.pt').relative_to(ROOT))]==read_json(REPORT/'plan.json')['source_sha256']
    os.environ['PIDG_METRIC_CONFIG']=str(SOURCE/'metric_config.json')
    g,d,proof=load(SOURCE)
    count=len(g.get_xyz);total=np.zeros((count,3),np.float64);votes=np.zeros(count,int)
    best_mass=np.zeros(count);best_frame=np.zeros(count,int);blue_by_frame=[];cache={};projection={};centers={};depths={}
    image_data={n:images(frames[n]) for n in numbers}
    semantic={n:images(frames[n],core=False) for n in numbers}
    for step,n in enumerate(numbers):
        cam=camera(frames[n],meta,1600);rgb,blue,yellow=image_data[n]
        _,full_blue,full_yellow=semantic[n]
        if n in clip_frames:
            masks=np.stack([*(components[n]==j for j in range(4)),full_blue,np.ones_like(blue),blue,np.zeros_like(blue),np.zeros_like(blue)])
        else:masks=np.stack([full_blue,full_yellow,np.ones_like(blue),blue,np.zeros_like(blue),np.zeros_like(blue)])
        masks=np.repeat(np.repeat(masks,4,1),4,2)
        mass,w,x,error=extract(g,d,cam,n,masks)
        if n in clip_frames:
            cache[n]=mass
            stat=np.stack([mass[:,4],mass[:,:4].sum(1),mass[:,5]],1)
            core_mass=mass[:,6]
        else:stat=mass[:,:3];core_mass=mass[:,3]
        total+=stat;votes+=stat[:,0]>=.02;blue_by_frame.append(core_mass)
        better=stat[:,0]>best_mass;best_mass[better]=stat[better,0];best_frame[better]=n
        with torch.no_grad():
            sm=camera(frames[n],meta,400)
            uv,cov=projected(x,g.get_scaling+w['d_scaling'],g.get_rotation+w['d_rotation'],sm)
            z=(x@sm.world_view_transform[:3,:3]+sm.world_view_transform[3,:3])[:,2]
        # Small projections are analytic coordinates of the native renderer.
        projection[n]=uv.cpu().numpy();depths[n]=z.cpu().numpy();centers[n]=x.cpu().numpy()
        if (step+1)%15==0:print('native contribution binding',step+1,'/120',flush=True)
    batches=[];bank_rows=[];dynamic_ids=set()
    for clip_index,clip in enumerate(clips):
        ns=clip['frames'];labels=components[ns[0]].copy();alive=np.ones(4,bool);mapping={}
        for j,n in enumerate(ns):
            if j:labels,_=propagate(labels,components[n],*flow[(ns[j-1],n)])
            mm,alive=qualify_fragments(labels,components[n],alive);assert alive.all()
            mapping[n]=mm
        banks=[]
        for group in range(4):
            good=np.ones(count,bool)
            for j,n in enumerate(ns[:3]):
                region=mapping[n][group];m=cache[n]
                good&=(m[:,region]>=.02)&(m[:,region]/np.maximum(m[:,5],1e-20)>=.95)&(depths[n]>.2)&inside(components[n]==region,projection[n])
                if j:
                    a=ns[j-1];f,b=flow[(a,n)];p=torch.tensor(projection[a])[:,None]
                    valid=sample(forward_backward_valid(f,b).float(),p)[:,0].numpy()>.99
                    target=projection[a]+sample(f,p)[:,0].numpy()
                    good&=valid&(np.linalg.norm(projection[n]-target,axis=1)<=2)
            ids=np.flatnonzero(good)
            if len(ids)<20:continue
            banks.append((group,ids));dynamic_ids.update(ids.tolist())
            bank_rows.append({'clip':clip['name'],'group':group,'points':len(ids)})
        assert len(banks)>=2, 'Too few independently supported local duck patches'
        ids=np.concatenate([x[1] for x in banks]);group=np.concatenate([np.full(len(x[1]),x[0]) for x in banks])
        uv=projection[ns[0]][ids].copy();live=np.ones(len(ids),bool)
        for j,n in enumerate(ns):
            if j:
                f,b=flow[(ns[j-1],n)];p=torch.tensor(uv)[:,None]
                valid=sample(forward_backward_valid(f,b).float(),p)[:,0].numpy()>.99
                uv=uv+sample(f,p)[:,0].numpy()
                live&=valid
            for gg in range(4):
                use=group==gg
                live[use]&=inside(ndimage.binary_erosion(components[n]==mapping[n][gg]),uv[use])
            use=live.copy()
            if use.sum()<20:continue
            edge=fixed_edges(centers[ns[0]],ids[use],group[use])
            batches.append({'frame':n,'clip':clip_index,'ids':ids[use].copy(),'target':uv[use].copy(),
                            'edges':edge,'local_labels':group[use].copy()})
    static=(total[:,0]/np.maximum(total[:,2],1e-20)>=.98)&(total[:,1]/np.maximum(total[:,2],1e-20)<=.005)&(votes>=3)&(g.get_opacity.detach().cpu().numpy()[:,0]>=.05)
    if dynamic_ids:static[list(dynamic_ids)]=False
    static_ids=np.flatnonzero(static)
    blue_by_frame=np.asarray(blue_by_frame)
    share=blue_by_frame[:,static].sum(1)/blue_by_frame.sum(1).clip(1e-10)
    thirds=[float(share[np.array([(lo<=n<=hi) for n in numbers])].mean()) for lo,hi in [(1,67),(68,134),(135,200)]]
    assert share.mean()>=.5 and min(thirds)>=.3, (share.mean(),thirds)
    anchor_ids=[];anchor_xyz=[];anchor_report=[]
    for track in read_json(REPORT/'static_image_audit.json')['tracks']:
        ns=track['frames'];target=np.asarray(track['uv'])
        source=ns[0]
        candidates=static_ids[np.linalg.norm(projection[source][static_ids]-target[0],axis=1)<=3]
        if not len(candidates):continue
        errors=np.stack([np.linalg.norm(projection[n][candidates]-p,axis=1) for n,p in zip(ns,target)])
        median=np.median(errors,0);rms=np.sqrt(np.mean(errors**2,0))
        good=(median<=2)&(rms<=3)
        if not good.any():continue
        ix=np.flatnonzero(good)[np.argmin(rms[good])];row=int(candidates[ix])
        anchor_ids.append(row);anchor_xyz.append(track['xyz'])
        anchor_report.append({'row':row,'source':source,'observations':len(ns),'source_median_reprojection_px':float(median[ix]),'source_RMS_px':float(rms[ix])})
    assert len(anchor_ids)>=8, f'Insufficient observed static3D anchors: {len(anchor_ids)}'
    arrays={'static_ids':static_ids,'static_reference_frames':best_frame[static],
            'anchor_ids':np.asarray(anchor_ids),'anchor_xyz':np.asarray(anchor_xyz,dtype=np.float32),
            'point_count':np.asarray(count),'source_checkpoint_sha256':np.asarray(proof['sha256']),
            'batch_count':np.asarray(len(batches))}
    for j,b in enumerate(batches):
        for key,value in b.items():arrays[f'b{j}_{key}']=np.asarray(value)
    RUNS.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(binding_path,**arrays)
    result={'path':str(binding_path),'sha256':sha256(binding_path),'source_checkpoint_sha256':proof['sha256'],
            'point_count':count,'static_points':len(static_ids),'static_mean_blue_alpha_share':float(share.mean()),
            'static_blue_share_time_thirds':thirds,'static_share_per_frame':dict(zip(map(str,numbers),map(float,share))),
            'independent_static_3D_anchors':len(anchor_ids),'anchors':anchor_report,'duck_local_banks':bank_rows,
            'local_observation_batches':len(batches),'local_point_observations':sum(len(b['ids']) for b in batches),
            'full_material_IDs':False,'new_view_dense_GT':False,'training_only':True}
    assert all(sha256(ROOT/p)==h for p,h in locked.items())
    write_json(REPORT/'input_lock.json',locked);write_json(REPORT/'data_binding.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['anchors','static_share_per_frame','duck_local_banks']},indent=2),flush=True)
    return result


if __name__=='__main__':
    torch.set_num_threads(4);prepare()
