"""Single serialized privileged cost audit; no reconstruction output allowed."""
import argparse
import msvcrt
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from ..prepare_data import ROOT,read,sha
from ..rgb_prior_proposals.io import atomic_json
from ..synthetic_audit.input import load
from ..pretrained_depth.model import configure
from ..adaptive_static.geometry import patch_features
from .prepare import BASE,REPORT,RUNS,OUT,D1,SOURCE,stopped,verify
from .core import single_face,world_points,scene_hit,nearest_grid,measure,true_visibility,controls
from .stats import summarize,groups,merge,compact


def write_new(path,data):
    assert not path.exists(),path
    atomic_json(path,data)


def save_or_verify(path,data):
    if path.exists():
        assert read(path)==data,'Saved aggregate must replay exactly: '+str(path)
    else:
        write_new(path,data)


def diagnostic(rgb,uv,costs,fit,accepted,single,visible,case,index):
    """Fixed view color visualization only; never a reusable GT-cost array."""
    h,w = rgb.shape[:2]; panels = [Image.fromarray((rgb*255).round().astype('uint8'))]
    colors = []
    face = np.where(single[:,None],[40,210,130],[240,160,30]).astype('uint8'); colors.append(face)
    vis = np.where(visible.all(0)[:,None],[40,210,130],np.where(((visible[:4].sum(0)>=3)&visible[4:].all(0))[:,None],[240,160,30],[170,65,210])).astype('uint8'); colors.append(vis)
    for hypothesis in [1,2]:
        delta = fit[hypothesis]-fit[0]
        magnitude = np.clip(np.abs(delta)/.1,0,1)
        c = np.zeros((len(uv),3),np.uint8)
        c[:,0] = np.where(delta>0,50+205*magnitude,35)
        c[:,2] = np.where(delta<0,50+205*magnitude,35)
        c[:,1] = 50; colors.append(c)
    colors.append(np.where(accepted[0,:,None],[40,210,130],[200,65,65]).astype('uint8'))
    for color in colors:
        im = np.zeros((h,w,3),np.uint8); im[uv[:,1],uv[:,0]] = color; panels.append(Image.fromarray(im))
    out = Image.new('RGB',(w*3,(h+34)*2+48),'#171b24'); draw = ImageDraw.Draw(out)
    titles = ['original RGB','single face green; mixed orange','visibility: all6 green; enough orange',
              'GT-D1 fit: red worse; blue better','gridGT-D1: red worse; blue better','GT acceptance: green pass; red reject']
    for j,(im,title) in enumerate(zip(panels,titles)):
        x,y=(j%3)*w,(j//3)*(h+34); draw.text((x+3,y+3),title,fill='white'); out.paste(im,(x,y+32))
    draw.text((4,(h+34)*2+3),f'{case} fixed view {index}; black = missing/not evaluated, not zero error. Evaluator only.',fill='white')
    path = OUT/f'{case}_view_{index:03d}.png'
    if path.exists():
        with Image.open(path) as saved: assert np.array_equal(np.asarray(saved),np.asarray(out))
    else: out.save(path)
    return path


@torch.no_grad()
def view_audit(values,features,texture,views,k,index,sources,near,far,gt,depth):
    yy,xx = np.nonzero(depth > 0); all_uv = np.c_[xx,yy]
    supported = gt['observed_static_visible'][yy,xx] & (gt['static_depth'][yy,xx] > 0)
    uv = all_uv[supported]; x,y=uv.T
    old,truth = depth[y,x],gt['static_depth'][y,x]
    assert len(uv)>0 and np.isfinite(old).all() and np.isfinite(truth).all()
    # Independent geometry sanity check, before using it to group source visibility.
    inverse = np.linalg.inv(values['views'][index]); origin = inverse[:3,3]
    rays = (np.c_[uv,np.ones(len(uv))] @ np.linalg.inv(values['K']).T) @ inverse[:3,:3].T
    analytic = scene_hit(origin,rays,float(values['times'][index]),True)
    assert np.max(np.abs(analytic-truth)) < 1e-6
    points = world_points(uv,truth,values['views'][index],values['K'])
    h,w=depth.shape
    visible = true_visibility(points,values['views'],values['times'],sources,views,k,h,w)
    single = single_face(gt['static_depth'],gt['static_normal'],gt['observed_static_visible'])[y,x]
    thin = gt['thin_static_visible'][y,x] # already intersected with reference-visible static support
    cs,fs,ac=[],[],[]
    for start in range(0,len(uv),256):
        sl=slice(start,start+256)
        xy=torch.as_tensor(uv[sl],dtype=torch.float32,device=views.device)
        z=torch.as_tensor(np.stack([old[sl],truth[sl]]),device=views.device)
        hypotheses=torch.cat([z,nearest_grid(z[1],near,far)[None]],0)
        costs,fit,accepted=measure(xy,hypotheses,features,texture,views,k,index,sources,True)
        cs.append(costs.cpu().numpy()); fs.append(fit.cpu().numpy()); ac.append(accepted.cpu().numpy())
    costs,fit,accepted=np.concatenate(cs,2),np.concatenate(fs,1),np.concatenate(ac,1)
    masks=groups(single,visible,np.abs(old-truth),thin)
    statistics={name:summarize(costs,fit,accepted,mask) for name,mask in masks.items()}
    counts={'D1_all_candidates':len(all_uv),'reference_visible_static_candidates':len(uv),
        'other_reference_unknown_candidates':len(all_uv)-len(uv),
        'all_visible_static_rays':int(gt['observed_static_visible'].sum()),
        'all_visible_thin_rays':int((gt['thin_static_visible']&gt['observed_static_visible']).sum()),
        'static_analytic_depth_max_abs_difference':float(np.max(np.abs(analytic-truth)))}
    return statistics,counts,(uv,costs,fit,accepted,single,visible)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--enable-photometric-oracle',action='store_true'); args=p.parse_args()
    if not args.enable_photometric_oracle: raise SystemExit('Privileged audit is opt-in; no files read or written')
    assert not stopped(); verify(); assert not (REPORT/'measurement.json').exists(),'Completed audit must not repeat'
    RUNS.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    lock=(RUNS/'audit.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    configure(); freeze=sha(REPORT/'source_freeze.json'); outputs={}; results={}
    try:
        control_path=REPORT/'analytic_controls.json'
        save_or_verify(control_path,{'controls':controls('cuda'),'evaluator_only':True})
        for case in ['diffuse','glossy']:
            train=ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train'
            values=load(train); meta=read(train/'cameras.json'); source=read(SOURCE/f'{case}_inference.json')
            near,far=source['camera_derived_depth_range']
            images=torch.from_numpy(values['rgb']).permute(0,3,1,2).cuda()
            features,texture=patch_features(images,5); del images
            views=torch.as_tensor(values['views'],dtype=torch.float32,device='cuda')
            k=torch.as_tensor(values['K'],dtype=torch.float32,device='cuda')
            rows=[]; counts=[]; per_view=[]
            for i,row in enumerate(meta['frames']):
                assert not stopped(),'User stop: retain all completed aggregate saves, release this task GPU'
                path=RUNS/case/f'view_{i:03d}.json'; gtpath=train.parent/'evaluation_only/gt'/f'{Path(row["image"]).stem}.npz'
                binding={'freeze_sha256':freeze,'D1_sha256':sha(D1/case/f'view_{i:03d}.npz'),
                         'GT_sha256':sha(gtpath),'sources':source['sources'][i],'case':case,'index':i}
                if path.exists():
                    saved=read(path); assert saved['bindings']==binding
                else:
                    with np.load(D1/case/f'view_{i:03d}.npz') as a: depth=a['refined'].copy()
                    with np.load(gtpath) as a: gt={key:a[key] for key in ['static_depth','static_normal','observed_static_visible','thin_static_visible']}
                    stat,count,display=view_audit(values,features,texture,views,k,i,source['sources'][i],near,far,gt,depth)
                    if i in [0,24,72]: diagnostic(values['rgb'][i],*display,case,i)
                    saved={'bindings':binding,'counts':count,'groups':stat}
                    write_new(path,saved)
                rows.append(saved['groups']); counts.append(saved['counts'])
                outputs[path.relative_to(ROOT).as_posix()]=sha(path)
                per_view.append({'index':i,'counts':saved['counts'],'groups':compact(saved['groups'])})
                atomic_json(RUNS/'state.json',{'status':'auditing','pid':os.getpid(),'case':case,'completed':i+1,'total':96})
                if i%12==0: print('PHOTOMETRIC_ORACLE',case,i+1,flush=True)
            combined=merge(rows)
            total={key:sum(c[key] for c in counts) for key in counts[0] if 'max_abs' not in key}
            total['static_analytic_depth_max_abs_difference']=max(c['static_analytic_depth_max_abs_difference'] for c in counts)
            results[case]={'counts':total,'groups':compact(combined)}
            save_or_verify(REPORT/f'{case}_per_view.json',{'views':per_view,'histogram_quantile_resolution_cost':.01,'difference':.02})
            save_or_verify(RUNS/f'{case}_aggregate_histograms.json',combined)
            del rows,features,texture,views,k; torch.cuda.empty_cache()
        verify()
        figures={p.relative_to(ROOT).as_posix():sha(p) for p in OUT.glob('*.png')}
        write_new(REPORT/'measurement.json',{'evaluator_only':True,'Gaussian_training':False,'results':results,
            'source_freeze_sha256':freeze,'outputs':outputs,'figures':figures,
            'group_contract':'all6 subset of fit3+held; cross groups exclusive; other reference candidates remain unknown',
            'quantiles':'histogram midpoint estimates; .01 cost / .02 difference bins',
            'no_per_pixel_GT_cost_depth_normal_mask_or_training_point_export':True})
        atomic_json(RUNS/'state.json',{'status':'measured_pending_review','views':192})
        print('PHOTOMETRIC_ORACLE_COMPLETE',flush=True)
    except Exception as error:
        atomic_json(RUNS/'state.json',{'status':'saved_stopped' if stopped() else 'failed','error':repr(error),'pid':os.getpid()})
        raise
    finally:
        torch.cuda.empty_cache(); lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1); lock.close()


if __name__ == '__main__': main()
