"""Independent process accounting / original-cost replay, no GT pixel exports."""
import numpy as np
import torch
from pathlib import Path
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha
from variants.staged_reconstruction.photometric_oracle.prepare import BASE,REPORT,RUNS,OUT,D1,SOURCE,verify
from variants.staged_reconstruction.photometric_oracle.stats import merge,compact,summarize
from variants.staged_reconstruction.synthetic_audit.input import load
from variants.staged_reconstruction.adaptive_static.geometry import patch_features,source_cost,project_torch
from variants.staged_reconstruction.pretrained_depth.model import configure
from variants.staged_reconstruction.synthetic_audit.generate import box,sphere


def replay(values,features,texture,views,k,i,sources,near,far,depth,gt):
    yy,xx=np.nonzero(depth>0)
    good=gt['observed_static_visible'][yy,xx]&(gt['static_depth'][yy,xx]>0)
    uv=np.c_[xx[good],yy[good]]; x,y=uv.T; old=depth[y,x]; truth=gt['static_depth'][y,x]
    inv=np.linalg.inv(values['views'][i]); rays=(np.c_[uv,np.ones(len(uv))]@np.linalg.inv(values['K']).T)@inv[:3,:3].T
    xyz=inv[:3,3]+rays*truth[:,None]
    vis=[]
    for j in sources:
        origin=np.linalg.inv(values['views'][j])[:3,3]; r=xyz-origin; t=float(values['times'][j])
        candidates=[box(origin,r,[-.8,-.75,-.6],[.8,-.45,.6])[0],
            box(origin,r,[-.2,-.45,-.2],[.2,.9,.2])[0],
            box(origin,r,[.42,-.45,-.32],[.46,.6,.32])[0],
            sphere(origin,r,[.8*np.sin(2*np.pi*t),.28,.7*np.cos(2*np.pi*t)],.25)[0],
            box(origin,r,[-.45+.6*t,.62,-.7],[-.2+.6*t,.85,-.45])[0]]
        hits=np.min(candidates,axis=0)
        margin=project_torch(torch.tensor(xyz,dtype=torch.float32,device='cuda')[None],views[j],k,*depth.shape)[1][0].cpu().numpy()
        vis.append((np.abs(hits-1)<=1e-4)&margin)
    vis=np.stack(vis)
    single=np.ones(len(uv),bool); n=gt['static_normal'][y,x]; h,w=depth.shape
    for dy in range(-2,3):
        for dx in range(-2,3):
            a,b=y+dy,x+dx; inside=(a>=0)&(a<h)&(b>=0)&(b<w); a=a.clip(0,h-1); b=b.clip(0,w-1)
            single &= inside&gt['observed_static_visible'][a,b]&(gt['static_depth'][a,b]>0)&((gt['static_normal'][a,b]*n).sum(1)>.99)
    costs=[]
    for start in range(0,len(uv),256):
        sl=slice(start,start+256); xy=torch.tensor(uv[sl],dtype=torch.float32,device='cuda')
        z=torch.tensor(np.stack([old[sl],truth[sl]]),device='cuda')
        grid=1/torch.linspace(1/near,1/far,192,device='cuda')
        which=((1/z[1]-1/near)/((1/far-1/near)/191)).round().long().clamp(0,191)
        z=torch.cat([z,grid[which][None]],0)
        rr=torch.cat([xy,torch.ones_like(xy[:,:1])],1)@torch.linalg.inv(k).T
        iv=torch.linalg.inv(views[i]); pts=(rr[None]*z[...,None])@iv[:3,:3].T+iv[:3,3]
        ref=features[i,:,xy[:,1].long(),xy[:,0].long()]
        cc=torch.stack([source_cost(pts,ref,features[j],texture[j],views[j],k) for j in sources],1)
        costs.append(cc.cpu().numpy())
    c=np.concatenate(costs,axis=2); f=np.sort(c[:,:4],axis=1)[:,:3].mean(1)
    accepted=[]
    grid=1/torch.linspace(1/near,1/far,192,device='cuda'); zz=torch.tensor(truth,device='cuda')
    gridtruth=grid[((1/zz-1/near)/((1/far-1/near)/191)).round().long().clamp(0,191)].cpu().numpy()
    for hyp,new in [(1,truth),(2,gridtruth)]:
        a=np.isfinite(new)&(new>0)&np.isfinite(old)&(old>0)&(new!=old)
        a&=(f[hyp]<=f[0]+1e-6)&(c[hyp,4:]<=c[0,4:]+1e-6).all(0)&(c[hyp,4:]<c[0,4:]-1e-6).any(0)
        accepted.append(a)
    a=np.stack(accepted); all6=vis.all(0); enough=(vis[:4].sum(0)>=3)&vis[4:].all(0)
    groups={'all':np.ones(len(uv),bool),'single_face':single,'mixed_edge_missing':~single,
        'all6_visible_subset':all6,'fit3_bothheld_visible_inclusive':enough,'other_visibility':~enough,
        'visible_thin':gt['thin_static_visible'][y,x]}
    error=np.abs(old-truth)
    for key,mask in [('le02',error<=.02),('02to05',(error>.02)&(error<=.05)),('gt05',error>.05)]:
        groups['error_'+key]=mask
        for fn,fm in [('single',single),('mixed',~single)]:
            for vn,vm in [('all6',all6),('enough_notall6',enough&~all6),('other',~enough)]: groups[fn+'/'+vn+'/'+key]=fm&vm&mask
    return {key:summarize(c,f,a,mask) for key,mask in groups.items()}


def main():
    configure(); verify(); measurement=read(REPORT/'measurement.json'); results={}
    for case in ['diffuse','glossy']:
        rows=[]; total=0
        for i in range(96):
            p=RUNS/case/f'view_{i:03d}.json'
            assert sha(p)==measurement['outputs'][p.relative_to(ROOT).as_posix()]
            saved=read(p); g=saved['groups']; n=g['all']['n']; cross=[v for key,v in g.items() if '/' in key]
            assert sum(x['n'] for x in cross)==n
            assert sum(x['continuous_accepted'] for x in cross)==g['all']['continuous_accepted']
            assert sum(x['grid_accepted'] for x in cross)==g['all']['grid_accepted']
            assert g['all6_visible_subset']['n']<=g['fit3_bothheld_visible_inclusive']['n']<=n
            assert g['single_face']['n']+g['mixed_edge_missing']['n']==n
            assert n==saved['counts']['reference_visible_static_candidates']
            for group in g.values():
                for name,m in group['metrics'].items():
                    assert sum(m['histogram'])==m['n']==group['n']*(6 if name.endswith('six_sources') else 1)
            total+=saved['counts']['D1_all_candidates']; rows.append(g)
        assert compact(merge(rows))==measurement['results'][case]['groups']
        train=ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train'; values=load(train); meta=read(train/'cameras.json')
        source=read(SOURCE/f'{case}_inference.json'); near,far=source['camera_derived_depth_range']
        features,texture=patch_features(torch.tensor(values['rgb'],device='cuda').permute(0,3,1,2),5)
        views=torch.tensor(values['views'],dtype=torch.float32,device='cuda'); k=torch.tensor(values['K'],dtype=torch.float32,device='cuda')
        for i in [0,24,72]:
            with np.load(D1/case/f'view_{i:03d}.npz') as a: depth=a['refined'].copy()
            with np.load(train.parent/'evaluation_only/gt'/f'{Path(meta["frames"][i]["image"]).stem}.npz') as a: gt={key:a[key] for key in ['static_depth','static_normal','observed_static_visible','thin_static_visible']}
            independent=replay(values,features,texture,views,k,i,source['sources'][i],near,far,depth,gt)
            saved=read(RUNS/case/f'view_{i:03d}.json')['groups']
            # Source-cost tensors are exact; NumPy vs torch fit reductions may differ by FP32 rounding.
            for key,g in independent.items():
                for field in ['n','continuous_accepted','grid_accepted','unit_source_costs','fit_nonworse','held_nonworse']:
                    assert g[field]==saved[key][field],(case,i,key,field,g[field],saved[key][field])
                for name,m in g['metrics'].items():
                    if 'fit' not in name:
                        assert m==saved[key]['metrics'][name],(case,i,key,name,'source costs must be exact')
                    assert abs(m['sum']-saved[key]['metrics'][name]['sum'])<=1e-4*max(m['n'],1),(case,i,key,name)
        results[case]={'aggregate_views_verified':96,'all_original_D1_candidates_counted':total,
            'independent_original_RGB_cost_and_groups_replay_views':[0,24,72],'independent_acceptance_and_visibility_counts_exact':True}
        del rows,features,texture,views,k; torch.cuda.empty_cache()
    verify()
    write(REPORT/'independent_audit.json',{'passed':True,'results':results,'measurement_sha256':sha(REPORT/'measurement.json'),
        'source_sha256':sha(Path(__file__)),'frozen_inputs_unchanged':True,'per_pixel_GT_assets_exported':False,
        'fit_mean_reduction_numeric_tolerance':1e-4,'source_cost_implementation':'unchanged original source_cost',
        'six_source_and_held_cost_statistics_exact':True,
        'GPU_training_state_or_native_render_claim':False})
    print('D5_INDEPENDENT_ACCOUNTING_AND_RGB_REPLAY_PASS',flush=True)


if __name__=='__main__': main()
