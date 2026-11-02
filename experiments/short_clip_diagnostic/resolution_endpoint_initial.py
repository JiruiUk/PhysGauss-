from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.short_clip_diagnostic.image_camera import REPORT,OUT,TRAIN,read,write,sha,image
from output.short_clip_diagnostic.model_check import CACHE,extract
from output.surface_correspondence.common import setup,render
from output.surface_correspondence.audit import center_inside
from variants.surface_correspondence.model import Config,particle_support,assess_cohort
from variants.shape_coherence.observations import camera
from variants.flow_comparison.temporal import warp_at
from output.surface_support.core import RenderProxy


def training_rgb(frame,width):
    with Image.open(TRAIN/(frame['file_path']+'.png')) as im:rgba=np.array(im.convert('RGBA'))/255
    premult=(rgba[...,:3]*rgba[...,3:4]*255).astype('uint8')
    return np.array(Image.fromarray(premult).resize((width,width*9//16)))


def main():
    torch.set_num_threads(4);r=read(REPORT/'model_results.json');meta=read(TRAIN/'transforms_train.json');lookup={int(Path(f['file_path']).stem):f for f in meta['frames']}
    with np.load(ROOT/'output/surface_support/image_tracks.npz') as z:components={int(n):c for n,c in zip(z['frames'],z['components'])}
    g,d,proxy=setup();folder=OUT/'native_gt_render';folder.mkdir(exist_ok=True);results=[];native={};scores=[]
    numbers=sorted({n for c in r['clips'] for n in c['frames'][-2:]})
    for n in numbers:
        cam=camera(lookup[n],meta,1600)
        with torch.no_grad():w=warp_at(d,proxy.get_xyz,(n-1)/199)
        _,blue=image(lookup[n]);small=np.stack([*(components[n]==j for j in range(4)),blue,np.ones((225,400),bool)])
        masks=np.repeat(np.repeat(small,4,axis=1),4,axis=2)
        mass,alpha,error=extract(proxy,cam,w,masks);native[n]={'mass':mass/16,'alpha':alpha.reshape(225,4,400,4).mean((1,3))}
        np.savez_compressed(CACHE/f'native{n:04d}.npz',**native[n])
        with torch.no_grad():rgb=render(proxy,cam,w,None)['render'].clamp(0,1).permute(1,2,0).cpu().numpy()
        pred=(rgb*255).round().astype('uint8');gt=training_rgb(lookup[n],1600)
        Image.fromarray(np.concatenate([gt,pred],axis=1)).save(folder/f'frame{n:04d}.png')
        with np.load(CACHE/f'frame{n:04d}.npz') as z:low=z['rgb']
        diagnostic_gt,_=image(lookup[n]);matched_gt=training_rgb(lookup[n],400);down=np.array(Image.fromarray(pred).resize((400,225)))/255
        def psnr(a,b):return float(-10*np.log10(np.mean((a-b)**2)))
        scores.append({'frame':n,'direct400_vs_diagnosticGT_PSNR':psnr(low,diagnostic_gt/255),'direct400_vs_trainingStyleGT_PSNR':psnr(low,matched_gt/255),
          'native1600_vs_trainingGT_PSNR':psnr(rgb,gt/255),'nativeRGB_down400_vs_trainingStyleGT_PSNR':psnr(down,matched_gt/255),
          'native_color_adjoint_integral_error':error})
        print('resolution frame',n,scores[-1],flush=True)
    config=Config(enabled=True)
    for clip in r['clips']:
        previous,target=clip['frames'][-2:];old={}
        for n in [previous,target]:
            with np.load(CACHE/f'frame{n:04d}.npz') as z:old[n]={k:z[k] for k in z.files}
        from output.short_clip_diagnostic.image_camera import FLOW
        from variants.flow_comparison.geometry import forward_backward_valid,sample
        row=next(x for x in read(FLOW/'manifest.json')['pairs'] if x['source']==previous and x['target']==target)
        with np.load(FLOW/row['file']) as z:f=torch.tensor(z['forward'],dtype=torch.float32);b=torch.tensor(z['backward'],dtype=torch.float32)
        p=torch.tensor(old[previous]['means']);q=torch.tensor(old[target]['means']);valid=sample(forward_backward_valid(f,b).float(),p[:,None])[:,0]>.99
        valid&=(p[:,0]>=0)&(p[:,0]<=399)&(p[:,1]>=0)&(p[:,1]<=224)
        e=torch.linalg.vector_norm(q-p-sample(f,p[:,None])[:,0],dim=-1).numpy();e[~valid.numpy()]=np.inf
        assignments={mode:np.full(len(p),-1,int) for mode in ['direct400','native1600_normalized']};banks=[]
        for bank in clip['banks']:
            if not bank['qualified'] or bank['name'].endswith('blue'):continue
            info=next(x for x in r['banks'] if x['name']==bank['name'])
            with np.load(ROOT/info['path']) as z:ix=z['kernel_ids']
            current_region=next(x['region'] for x in r['bank_frames'] if x['clip']==clip['clip'] and x['frame']==target and x['bank']==bank['name'])
            source_region=next(x['region'] for x in r['bank_frames'] if x['clip']==clip['clip'] and x['frame']==previous and x['bank']==bank['name'])
            inside=center_inside(old[target]['means'][ix],old[target]['depth'][ix],old[target]['cov'][ix],components[target]==current_region)
            source_inside=center_inside(old[previous]['means'][ix],old[previous]['depth'][ix],old[previous]['cov'][ix],components[previous]==source_region)
            rr={'bank':bank['name']}
            for mode in assignments:
                tm=old[target]['mass'] if mode=='direct400' else native[target]['mass'];sm=old[previous]['mass'] if mode=='direct400' else native[previous]['mass']
                score=assess_cohort(tm[ix,current_region],tm[ix,5],inside,e[ix],True,config)
                good=particle_support(tm[ix,current_region],tm[ix,5],inside,e[ix],config)&particle_support(sm[ix,source_region],sm[ix,5],source_inside,None,config)
                if not score['accepted']:good[:]=False
                assignments[mode][ix[good]]=current_region
                rr[mode]={'score':score,'accepted_particles':int(good.sum())}
            banks.append(rr)
        coverage={}
        for mode,a in assignments.items():
            mm=old[target]['mass'] if mode=='direct400' else native[target]['mass'];alpha=old[target]['alpha'] if mode=='direct400' else native[target]['alpha']
            selected=np.where(a>=0)[0];coverage[mode]=float(mm[selected,a[selected]].sum()/alpha[components[target]>=0].sum())
        results.append({'clip':clip['clip'],'source':previous,'target':target,'banks':banks,'endpoint_joint_coverage':coverage})
    write(REPORT/'resolution_control.json',{'RGB':scores,'endpoints':results,'same_banks_and_thresholds':True,'no_training':True,
       'mask_note':'Same400 masks, nearest upsampled; no independent native-resolution segmentation',
       'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},})
    print('RESOLUTION CONTROL COMPLETE',flush=True)


if __name__=='__main__':main()
