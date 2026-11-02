"""Frozen C1 local banks and controlled center/footprint/occlusion probes."""
from pathlib import Path
import sys,time
import numpy as np
import torch
from scipy import ndimage
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.short_clip_diagnostic.image_camera import REPORT,OUT,TRAIN,FLOW,read,write,sha,image
from output.short_clip_diagnostic.camera_math import w2c,intrinsic,project,motion_split
from output.surface_correspondence.common import setup,geometry,render,DIRECTORY
from output.surface_support.core import propagate,RenderProxy
from output.surface_correspondence.audit import center_inside,flow_at_centers
from variants.surface_correspondence.model import Config,qualify_fragments,particle_support,freeze_bank,assess_cohort
from variants.flow_comparison.geometry import sample
CONFIG=Config(enabled=True);CACHE=ROOT/'runs/pidg_short_clip_diagnostic/cache'


def extract(proxy,cam,warp,masks):
    masks=torch.as_tensor(masks,device='cuda',dtype=torch.float32);parts=[];errors=[];alpha=None
    for start in [0,3]:
        color=torch.ones(len(proxy.get_xyz),3,device='cuda',requires_grad=True);p=render(proxy,cam,warp,color);alpha=p['alpha'].squeeze().detach()
        assert float((p['render'].detach()-alpha[None]).abs().max())<1e-6
        grad=torch.autograd.grad((p['render']*masks[start:start+3]).sum(),color)[0].detach()
        expected=(alpha[None]*masks[start:start+3]).sum((1,2));error=float(((grad.sum(0)-expected).abs()/expected.clamp_min(1)).max())
        assert error<5e-4 and torch.isfinite(grad).all() and float(grad.min())>=-1e-7
        parts.append(grad.cpu().numpy());errors.append(error);del p,grad,color
    return np.concatenate(parts,1),alpha.cpu().numpy(),max(errors)


def main():
    torch.set_num_threads(4);CACHE.mkdir(parents=True,exist_ok=True);(OUT/'gt_render').mkdir(parents=True,exist_ok=True);(OUT/'banks').mkdir(exist_ok=True)
    parent=read(REPORT/'input_lock.json');own=[Path(__file__),ROOT/'output/surface_correspondence/common.py',ROOT/'output/surface_correspondence/audit.py']
    lock={**parent,**{str(p.relative_to(ROOT)):sha(p) for p in own}}
    assert all(sha(ROOT/p)==v for p,v in lock.items());write(REPORT/'model_input_lock.json',lock)
    clips=read(ROOT/'experiments/short_clip_audit/results.json')['selected'];metadata=read(TRAIN/'transforms_train.json');lookup={int(Path(f['file_path']).stem):f for f in metadata['frames']}
    zz=np.load(ROOT/'output/surface_support/image_tracks.npz');component={int(n):c for n,c in zip(zz['frames'],zz['components'])}
    manifest=read(FLOW/'manifest.json');pairs={}
    for row in manifest['pairs']:
        if any(row['source'] in c['frames'] and row['target'] in c['frames'] for c in clips):
            with np.load(FLOW/row['file']) as f:pairs[(row['source'],row['target'])]=(torch.tensor(f['forward'],dtype=torch.float32),torch.tensor(f['backward'],dtype=torch.float32))
    numbers=sorted({n for c in clips for n in c['frames']});g,d,proxy=setup();canonical=proxy.get_xyz.cpu().numpy()
    data={};blues={};frame_rows=[];k=intrinsic(metadata['camera_angle_x'],'legacy');background_before=sha(DIRECTORY/'checkpoints/resume.pt')
    for number in numbers:
        gt,blue=image(lookup[number]);blues[number]=blue;cam,w,c,p,cov,depth=geometry(g,d,lookup[number],metadata)
        masks=np.stack([*(component[number]==j for j in range(4)),blue,np.ones((225,400),bool)])
        mass,alpha,error=extract(proxy,cam,w,masks)
        with torch.no_grad():normal=render(proxy,cam,w,None)['render'].clamp(0,1).permute(1,2,0).cpu().numpy()
        view=w2c(lookup[number]['transform_matrix'],'legacy');view_error=float(np.max(np.abs(view-cam.world_view_transform.T.cpu().numpy())))
        assert view_error<3e-5
        cpu_uv,_=project(c.cpu().numpy(),view,k);visible=depth.cpu().numpy()>.2;projection_error=float(np.max(np.abs(cpu_uv[visible]-p.cpu().numpy()[visible])))
        assert projection_error<.003
        data[number]={'centers':c.cpu().numpy(),'means':p.cpu().numpy(),'cov':cov.cpu().numpy(),'depth':depth.cpu().numpy(),'mass':mass,'alpha':alpha,'rgb':normal,
            'scales':(g.get_scaling+w['d_scaling']).detach().cpu().numpy(),'rotations':(g.get_rotation+w['d_rotation']).detach().cpu().numpy()}
        np.savez_compressed(CACHE/f'frame{number:04d}.npz',**data[number])
        pred=(normal*255).round().astype('uint8');Image.fromarray(np.concatenate([gt,pred],1)).save(OUT/'gt_render'/f'frame{number:04d}.png')
        diff=normal-gt/255;duck=component[number]>=0
        frame_rows.append({'frame':number,'full_PSNR':float(-10*np.log10(np.mean(diff**2))),'duck_ROI_MAE':float(np.abs(diff[duck]).mean()),
            'color_adjoint_integral_relative_error':error,'legacy_view_max_error':view_error,'center_projection_error_pixels':projection_error})
        print('C1 frame',number,'duckMAE',frame_rows[-1]['duck_ROI_MAE'],flush=True)
    flow_errors={};observed={}
    for source,target in sorted(pairs):
        flow_errors[(source,target)]=flow_at_centers(torch.tensor(data[source]['means'],device='cuda'),torch.tensor(data[target]['means'],device='cuda'),pairs[(source,target)])
        observed[(source,target)]=sample(pairs[(source,target)][0].cuda(),torch.tensor(data[source]['means'],device='cuda')[:,None])[:,0].cpu().numpy()
    summaries=[];all_banks=[];bank_cache={};counter=[];bank_frame_rows=[];assignment_arrays={}
    for clip in clips:
        start=clip['frames'][0];ids=clip['frames'];labels=component[start].copy();alive=np.ones(4,bool);mapping={}
        for index,number in enumerate(ids):
            if index:labels,_=propagate(labels,component[number],*pairs[(ids[index-1],number)])
            mm,alive=qualify_fragments(labels,component[number],alive);assert alive.all();mapping[number]=mm
        if start not in bank_cache:
            banks=[]
            for seed in range(5):
                supports=[]
                for index,number in enumerate(ids[:3]):
                    vv=data[number];region=int(mapping[number][seed]) if seed<4 else 4;mask=component[number]==region if seed<4 else blues[number]
                    inside=center_inside(vv['means'],vv['depth'],vv['cov'],mask)
                    error=None if index==0 else flow_errors[(ids[index-1],number)]
                    supports.append(particle_support(vv['mass'][:,region],vv['mass'][:,5],inside,error,CONFIG))
                bank=freeze_bank(supports,canonical,CONFIG);name=f'A{start}_'+(f'F{seed}' if seed<4 else 'blue')
                if bank is None:banks.append({'name':name,'seed':seed,'qualified':False,'seed_count':int(np.logical_and.reduce(supports).sum())});continue
                path=OUT/'banks'/f'{name}.npz';np.savez_compressed(path,**bank,seed_frames=ids[:3])
                banks.append({'name':name,'seed':seed,'qualified':True,'seed_count':len(bank['kernel_ids']),'rows':bank['kernel_ids'],'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
            bank_cache[start]=banks
        banks=bank_cache[start];rows=[];coverage=[];assigned=np.full((len(ids),len(canonical)),-1,dtype=np.int8)
        for index,number in enumerate(ids):
            vv=data[number];accepted_banks=0
            for bank in banks:
                if not bank['qualified']:continue
                ix=bank['rows'];seed=bank['seed'];region=int(mapping[number][seed]) if seed<4 else 4;mask=component[number]==region if seed<4 else blues[number]
                inside=center_inside(vv['means'][ix],vv['depth'][ix],vv['cov'][ix],mask)
                error=np.full(len(ix),np.inf) if index==0 else flow_errors[(ids[index-1],number)][ix]
                score=assess_cohort(vv['mass'][ix,region],vv['mass'][ix,5],inside,error,True,CONFIG)
                phase='seed' if index<3 else 'forward';score['accepted']=score['accepted'] and phase=='forward'
                good=particle_support(vv['mass'][ix,region],vv['mass'][ix,5],inside,error,CONFIG)
                if index:
                    previous=ids[index-1];pv=data[previous];pr=int(mapping[previous][seed]) if seed<4 else 4;pm=component[previous]==pr if seed<4 else blues[previous]
                    pi=center_inside(pv['means'][ix],pv['depth'][ix],pv['cov'][ix],pm)
                    good&=particle_support(pv['mass'][ix,pr],pv['mass'][ix,5],pi,None,CONFIG)
                else:good[:]=False
                if not score['accepted']:good[:]=False
                if score['accepted'] and seed<4:assigned[index,ix[good]]=region;accepted_banks+=int(good.any())
                weights=vv['mass'][ix,region];wm=float(weights.sum());displacement=np.linalg.norm(vv['centers'][ix]-data[start]['centers'][ix],axis=1)
                split={}
                if index:
                    previous=ids[index-1];full,cam_only,obj=motion_split(data[previous]['centers'][ix],vv['centers'][ix],w2c(lookup[previous]['transform_matrix']),w2c(lookup[number]['transform_matrix']),k)
                    assert np.max(np.abs(full-cam_only-obj))<1e-8
                    target=observed[(previous,number)][ix];valid=np.isfinite(error)&inside
                    for key,value in [('full_projection',full),('camera_only',cam_only)]:
                        e=np.linalg.norm(value-target,axis=1);split[key+'_weighted_EPE']=float(np.sum(e[valid]*weights[valid])/weights[valid].sum()) if weights[valid].sum()>1e-12 else None
                    split['model_world_motion_projected_weighted_pixels']=float(np.sum(np.linalg.norm(obj[valid],axis=1)*weights[valid])/weights[valid].sum()) if weights[valid].sum()>1e-12 else None
                rows.append({'frame':number,'bank':bank['name'],'region':region,'phase':phase,'score':score,'accepted_points':int(good.sum()),
                    'own_alpha_coverage':float(wm/vv['alpha'][mask].sum()),'world_center_displacement_from_seed_median_scene_units':float(np.median(displacement)),
                    'weighted_raw_center_EPE':float(np.sum(error[np.isfinite(error)]*weights[np.isfinite(error)])/weights[np.isfinite(error)].sum()) if weights[np.isfinite(error)].sum()>1e-12 else None,
                    'motion_split':split})
            own=0.
            for region in range(4):own+=float(vv['mass'][np.where(assigned[index]==region)[0],region].sum())
            coverage.append({'frame':number,'phase':'seed' if index<3 else 'forward','accepted_duck_banks':accepted_banks,'accepted_duck_points':int((assigned[index]>=0).sum()),
                'accepted_duck_alpha_coverage':own/float(vv['alpha'][component[number]>=0].sum())})
        assignment_arrays[clip['name']]=assigned
        # Four controlled probes at the clip end; old model/camera unchanged.
        number=ids[-1];cam,w,_,_,_,_=geometry(g,d,lookup[number],metadata)
        for bank in banks:
            if not bank['qualified']:continue
            ix=bank['rows'];seed=bank['seed'];region=int(mapping[number][seed]) if seed<4 else 4;mask=component[number]==region if seed<4 else blues[number]
            selector=torch.zeros(len(canonical),1,device='cuda');selector[torch.tensor(ix,device='cuda')]=1
            isolated=RenderProxy(proxy,get_opacity=proxy.get_opacity*selector);white=torch.ones(len(canonical),3,device='cuda')
            fixed=RenderProxy(isolated,get_scaling=torch.tensor(data[start]['scales'],device='cuda'),get_rotation=torch.tensor(data[start]['rotations'],device='cuda'))
            frozen_center=RenderProxy(isolated,get_xyz=torch.tensor(data[start]['centers'],device='cuda'))
            identity=torch.zeros_like(w['d_xyz']);identity[:,0]=1
            with torch.no_grad():
                actual=render(isolated,cam,w,white)['render'][0].cpu().numpy()
                wf={**w,'d_rotation':torch.zeros_like(w['d_rotation']),'d_scaling':torch.zeros_like(w['d_scaling'])}
                footprint=render(fixed,cam,wf,white)['render'][0].cpu().numpy()
                center=render(frozen_center,cam,{**w,'d_xyz':identity},white)['render'][0].cpu().numpy()
            counter.append({'clip':clip['name'],'frame':number,'bank':bank['name'],'region':region,'normal_scene_own_mass':float(data[number]['mass'][ix,region].sum()),
                'isolated_actual_mass':float(actual[mask].sum()),'isolated_seed_footprint_mass':float(footprint[mask].sum()),'isolated_seed_world_center_mass':float(center[mask].sum()),
                'counterfactual_is_not_GT':True})
        for row in rows:row['clip']=clip['name']
        bank_frame_rows.extend(rows);forward=coverage[3:]
        summaries.append({'clip':clip['name'],'frames':ids,'seed_frames':ids[:3],'banks':[{'name':b['name'],'qualified':b['qualified'],'seed_count':b['seed_count']} for b in banks],
            'forward_observations':len(forward),'mean_forward_accepted_duck_alpha_coverage':float(np.mean([r['accepted_duck_alpha_coverage'] for r in forward])),
            'accepted_duck_bank_frame_events':sum(r['accepted_duck_banks'] for r in forward),'coverage':coverage,
            'mean_duck_ROI_MAE':float(np.mean([r['duck_ROI_MAE'] for r in frame_rows if r['frame'] in ids]))})
        print('clip completed',clip['name'],summaries[-1]['mean_forward_accepted_duck_alpha_coverage'],flush=True)
    np.savez_compressed(OUT/'assignments.npz',**assignment_arrays)
    all_banks=[{k:v for k,v in b.items() if k!='rows'} for bb in bank_cache.values() for b in bb]
    write(REPORT/'model_results.json',{'clips':summaries,'frames':frame_rows,'bank_frames':bank_frame_rows,'banks':all_banks,'counterfactuals':counter,
        'checkpoint_sha256':background_before,'normal_model_camera_convention':'legacy trained convention; alternate hypotheses not substituted into frozen C1',
        'cache_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(CACHE.glob('frame*.npz'))},'unique_original_training_frames':len(numbers),'new_Gaussian_optimizer_updates':0})
    assert sha(DIRECTORY/'checkpoints/resume.pt')==background_before
    assert all(sha(ROOT/p)==v for p,v in lock.items());print('FROZEN C1 SHORT CLIP CHECK COMPLETE',flush=True)


if __name__=='__main__':main()
