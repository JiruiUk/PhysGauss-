"""Frozen local surface banks with forward-only validation and unknown retention."""
from pathlib import Path
import sys,time
import numpy as np
import torch
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.surface_correspondence.common import *
from output.surface_support.core import propagate
from variants.surface_correspondence.model import Config,particle_support,freeze_bank,assess_cohort,qualify_fragments
from variants.flow_comparison.geometry import sample,forward_backward_valid
CONFIG=Config(enabled=True)


def flow_at_centers(previous,current,pair):
    forward,backward=pair;forward=forward.cuda();backward=backward.cuda()
    points=previous[:,None,:]
    sampled=sample(forward,points)[:,0]
    valid=sample(forward_backward_valid(forward,backward).float(),points)[:,0]>.99
    valid&=(previous[:,0]>=0)&(previous[:,0]<=399)&(previous[:,1]>=0)&(previous[:,1]<=224)
    result=torch.linalg.vector_norm(current-previous-sampled,dim=-1)
    return torch.where(valid,result,torch.full_like(result,float('inf'))).cpu().numpy()


def center_inside(mean,depth,cov,component_mask):
    mask=ndimage.binary_erosion(component_mask,iterations=1)
    xy=np.rint(mean).astype(np.int64);inside=(depth>.2)&np.isfinite(mean).all(-1)&np.isfinite(cov).all((1,2))
    inside&=(cov[:,0,0]>0)&(np.linalg.det(cov)>0)
    inside&=(xy[:,0]>=0)&(xy[:,0]<400)&(xy[:,1]>=0)&(xy[:,1]<225)
    good=inside.copy();good[inside]&=mask[xy[inside,1],xy[inside,0]]
    return good


def tracklets(frames,components,pairs,anchors):
    mappings=np.full((len(anchors),len(frames),4),-1,dtype=np.int16);masks=[];records=[]
    for a,number in enumerate(anchors):
        start=int(np.where(frames==number)[0][0]);labels=components[start].copy();alive=np.ones(4,bool)
        if int(components[start].max())!=3:raise ValueError('Anchor not a separated four-component training image')
        local=np.full_like(components,-1,dtype=np.int8)
        ends=[None]*4
        for i in range(start,len(frames)):
            if i>start:labels,_=propagate(labels,components[i],*pairs[(int(frames[i-1]),int(frames[i]))])
            before=alive.copy();mapping,alive=qualify_fragments(labels,components[i],alive)
            for k in range(4):
                if before[k] and not alive[k]:ends[k]=int(frames[i])
                labels[labels==k]=k if alive[k] else -1
            mappings[a,i]=mapping;local[i]=labels.astype(np.int8)
        masks.append(local);records.append({'anchor':number,'start_index':start,'seed_frames':[int(n) for n in frames[start:start+3]],
             'first_unverified_frame_per_seed':ends,'local_seed_is_not_material_ID':True})
    np.savez_compressed(OUT/'fragment_tracks.npz',frames=frames,labels=np.array(masks),component_mapping=mappings)
    write(REPORT/'image_fragments.json',records)
    return mappings,records


def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    plan=read(REPORT/'plan.json');anchors=plan['anchor_original_training_frames']
    old=read(ROOT/'experiments/motion_correspondence/input_lock.json')
    paths=[Path(__file__),ROOT/'output/surface_correspondence/common.py',ROOT/'variants/surface_correspondence/model.py',
           ROOT/'tests/test_surface_correspondence.py',REPORT/'plan.json',REPORT/'renderer_contracts.json']
    lock={**old,**{str(p.relative_to(ROOT)):sha(p) for p in paths}}
    assert all(sha(ROOT/p)==v for p,v in lock.items());write(REPORT/'input_lock.json',lock)
    contracts=read(REPORT/'renderer_contracts.json');assert contracts['passed']
    assert all(sha(ROOT/p)==v for p,v in contracts['source_sha256'].items())
    z=np.load(ROOT/'output/surface_support/image_tracks.npz');frames=z['frames'];components=z['components']
    manifest=read(FLOW/'manifest.json');pairs={}
    for row in manifest['pairs']:
        with np.load(FLOW/row['file']) as f:pairs[(row['source'],row['target'])]=(torch.from_numpy(f['forward'].astype(np.float32)),torch.from_numpy(f['backward'].astype(np.float32)))
    mapping,fragments=tracklets(frames,components,pairs,anchors)
    g,d,proxy=setup();metadata=read(TRAIN/'transforms_train.json');lookup={int(Path(f['file_path']).stem):f for f in metadata['frames']}
    canonical=g.get_xyz.detach().cpu().numpy();seed_cache={}
    # Only15 predeclared seed observations; no future observations select rows.
    seed_numbers=sorted({number for fragment in fragments for number in fragment['seed_frames']})
    for number in seed_numbers:
        i=int(np.where(frames==number)[0][0]);cam,w,c,p,cov,depth=geometry(g,d,lookup[number],metadata)
        mass,alpha,error=full_particle_mass(proxy,cam,w,components[i])
        seed_cache[number]={'mass':mass.cpu().numpy(),'means':p.cpu().numpy(),'depth':depth.cpu().numpy(),'cov':cov.cpu().numpy()}
        np.savez_compressed(CACHE/f'seed{number:04d}.npz',**seed_cache[number]);print('seed extraction',number,flush=True)
    banks=[];rejected=[]
    for a,fragment in enumerate(fragments):
        start=fragment['start_index']
        for k in range(4):
            supports=[]
            for offset,number in enumerate(fragment['seed_frames']):
                i=start+offset;value=seed_cache[number];component=int(mapping[a,i,k]);own=np.zeros(len(canonical)) if component<0 else value['mass'][:,component]
                inside=center_inside(value['means'],value['depth'],value['cov'],components[i]==component) if component>=0 else np.zeros(len(canonical),bool)
                error=None
                if offset:
                    previous=seed_cache[fragment['seed_frames'][offset-1]]['means']
                    error=flow_at_centers(torch.as_tensor(previous,device='cuda'),torch.as_tensor(value['means'],device='cuda'),pairs[(int(frames[i-1]),number)])
                supports.append(particle_support(own,value['mass'][:,4],inside,error,CONFIG))
            bank=freeze_bank(supports,canonical,CONFIG);name=f'A{a}_F{k}'
            if bank is None:
                rejected.append({'name':name,'anchor':anchors[a],'local_seed':k,'three_frame_particle_intersection':int(np.logical_and.reduce(supports).sum()),'reason':'fewer_than20_jointly_supported_particles'});continue
            ids=bank['kernel_ids'];path=OUT/'banks'/f'{name}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,**bank,anchor=anchors[a],seed_frames=fragment['seed_frames'])
            banks.append({'name':name,'anchor':anchors[a],'anchor_index':a,'local_seed':k,'seed_frames':fragment['seed_frames'],
                 'seed_end_index':start+2,'particle_count':len(ids),'bank_file':str(path.relative_to(ROOT)),'bank_sha256':sha(path),'kernel_ids':ids})
            print('frozen bank',name,len(ids),flush=True)
    assert banks,'No bank qualifies; preserve rejection result before further work'
    union=np.unique(np.concatenate([b['kernel_ids'] for b in banks]));positions={int(k):i for i,k in enumerate(union)}
    ids_gpu=torch.as_tensor(union,device='cuda');rows=[];all_centers=[];all_means=[];all_areas=[];all_masses=[];all_errors=[];all_depth=[];cache_receipts=[]
    previous=None;began=time.time()
    for i,number0 in enumerate(frames):
        number=int(number0);cam,w,c,p,cov,depth=geometry(g,d,lookup[number],metadata)
        mass,alpha,numeric=full_particle_mass(proxy,cam,w,components[i])
        c=c[ids_gpu].cpu().numpy();p=p[ids_gpu].cpu().numpy();cov=cov[ids_gpu].cpu().numpy();depth=depth[ids_gpu].cpu().numpy();mm=mass[ids_gpu].cpu().numpy()
        error=np.full(len(union),np.inf) if previous is None else flow_at_centers(torch.as_tensor(previous,device='cuda'),torch.as_tensor(p,device='cuda'),pairs[(int(frames[i-1]),number)])
        bank_rows=[]
        for bank in banks:
            local=np.array([positions[int(k)] for k in bank['kernel_ids']]);component=int(mapping[bank['anchor_index'],i,bank['local_seed']])
            own=np.zeros(len(local)) if component<0 else mm[local,component]
            inside=center_inside(p[local],depth[local],cov[local],components[i]==component) if component>=0 else np.zeros(len(local),bool)
            result=assess_cohort(own,mm[local,4],inside,error[local],component>=0,CONFIG)
            phase='prebirth' if i<bank['seed_end_index']-2 else ('seed' if i<=bank['seed_end_index'] else 'forward_validation')
            if phase!='forward_validation':result['accepted']=False
            denominator=float(alpha[torch.as_tensor(components[i]==component,device='cuda')].sum()) if component>=0 else 0.
            result.update({'bank':bank['name'],'phase':phase,'component':component,'visible_fragment_alpha_mass':denominator,
                'fragment_alpha_coverage':float(own.sum()/denominator) if denominator>1e-12 else None,
                'all_duck_mass':float(mm[local,:4].sum()),'center_inside_unweighted_fraction':float(inside.mean())})
            bank_rows.append(result)
        all_centers.append(c);all_means.append(p);all_areas.append(np.sqrt(np.maximum(np.linalg.det(cov),0)));all_masses.append(mm);all_errors.append(error);all_depth.append(depth)
        rows.append({'frame':number,'numerical_adjoint_relative_error':max(numeric),'all_duck_alpha_mass':float(alpha[torch.as_tensor(components[i]>=0,device='cuda')].sum()),'banks':bank_rows})
        previous=p
        if i%10==0:print('full-sequence support',number,'accepted',sum(r['accepted'] for r in bank_rows),flush=True)
    np.savez_compressed(OUT/'surface_trajectories.npz',frames=frames,kernel_ids=union,canonical=canonical[union],centers=np.array(all_centers),
        projected=np.array(all_means),footprint_sqrt_det=np.array(all_areas),mass=np.array(all_masses),raw_flow_error=np.array(all_errors),depth=np.array(all_depth))
    overlap=[]
    for i,a in enumerate(banks):
        for b in banks[i+1:]:
            shared=len(np.intersect1d(a['kernel_ids'],b['kernel_ids']));den=len(np.union1d(a['kernel_ids'],b['kernel_ids']))
            overlap.append({'a':a['name'],'b':b['name'],'shared_canonical_rows':shared,'jaccard':shared/den,'physical_identity_merge_accepted':False})
    for bank in banks:del bank['kernel_ids']
    report={'banks':banks,'rejected_banks':rejected,'frames':rows,'anchor_bank_overlap_hypotheses':overlap,
       'all_original_training_frames':120,'selected_canonical_particle_union':len(union),'Gaussian_optimizer_updates':0,
       'new_trainable_parameters':0,'no_global_object_or_material_identity_claim':True,'no_cross_occlusion_identity_merge':True,
       'forward_validation_elapsed_seconds':time.time()-began,'all_SHA_unchanged':all(sha(ROOT/p)==v for p,v in lock.items()),
       'cache_receipts':{str(p.relative_to(ROOT)):sha(p) for p in sorted(CACHE.glob('seed*.npz'))}}
    assert report['all_SHA_unchanged'];write(REPORT/'results.json',report)
    print('SURFACE BANK AUDIT COMPLETE',len(banks),len(rejected),len(union),flush=True)


if __name__=='__main__':main()
