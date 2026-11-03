"""Read-only, final full-state U0/U1 evaluation; no novel-view targets."""
from pathlib import Path
import importlib.util
import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('transfer_render',ROOT/'output/fixed_camera_temporal/render_sequences.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.OUT=ROOT/'output/transfer_integrity';r.REPORT=ROOT/'experiments/transfer_integrity'
r.MODELS={'C1_original_50k':ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck',
          **{n:ROOT/'runs/pidg_transfer_integrity/pilot'/n/'rubber_duck' for n in ['U0_C1_control','U1_transfer_geometry']}}
from variants.metric_tuning.runtime import read_json,write_json,sha256
from variants.transfer_integrity.run import freeze,prepare_code


def select(directory):
    os.environ['PIDG_METRIC_CONFIG']=str(directory/'metric_config.json')
    from variants.color_repair.color import _read
    _read.cache_clear()
    return r.load(directory)


def training_proof():
    freeze(prepare_code())
    a,b=[r.MODELS[n] for n in ['U0_C1_control','U1_transfer_geometry']]
    configs=[read_json(d/'metric_config.json') for d in [a,b]]
    assert {k for k in configs[0] if configs[0][k]!=configs[1][k]}=={'transfer_integrity_enabled'}
    samples=[[read_json_line(x) for x in (d/'camera_sampling.jsonl').read_text().splitlines()] for d in [a,b]]
    assert samples[0]==samples[1] and len(samples[0])==2000
    assert [x['iteration'] for x in samples[0]]==list(range(50001,52001))
    losses=[read_json_line(x) for x in (b/'transfer_loss.jsonl').read_text().splitlines()]
    assert len(losses)==200 and all(np.isfinite(x['weighted_loss']) for x in losses)
    assert not (a/'transfer_loss.jsonl').exists()
    states={}
    for n,directory in r.MODELS.items():
        g,d,proof=select(directory)
        assert proof['iteration']==(50000 if n=='C1_original_50k' else 52000)
        states[n]=proof
        del g,d;torch.cuda.empty_cache()
    write_json(r.REPORT/'training_verification.json',{'states':states,'same_2000_camera_samples':True,
               'only_main_flag_differs':True,'new_geometry_loss_records':200,'frozen_inputs_and_code_unchanged':True,
               'source_C1_unchanged':states['C1_original_50k']['sha256']==read_json(r.REPORT/'plan.json')['source_sha256']})


def read_json_line(line):
    import json
    return json.loads(line)


@torch.no_grad()
def observations():
    from utils.loss_utils import ssim
    from pytorch_msssim import ms_ssim
    from variants.metric_tuning.model import quantize_png
    from variants.view_completeness.metrics import observed_coverage
    import lpips
    metric=lpips.LPIPS(net='vgg').cuda().eval().requires_grad_(False)
    keys=['PSNR','SSIM','LPIPS','MS-SSIM','blue_PSNR','duck_PSNR','blue_alpha_below09','duck_alpha_below09']
    for name,directory in r.MODELS.items():
        g,d,proof=select(directory)
        for split,expected in [('val',30),('test',50)]:
            path=r.REPORT/f'{name}_{split}.json'
            if path.exists():
                old=read_json(path)
                if old['full_state']==proof and old['evaluator_sha256']==sha256(Path(__file__)):continue
            meta=read_json(ROOT/f'data/derived/kinematic_resplit/transforms_{split}.json');rows=[]
            assert len(meta['frames'])==expected
            for f in meta['frames']:
                n=int(Path(f['file_path']).stem)
                with Image.open(ROOT/f'data/PIDG/rubber_duck/train/{n:04d}.png') as im:
                    rgba=np.asarray(im.convert('RGBA'),dtype=np.float32)/255
                # Match historical full-state evaluator: composite, quantize,
                # then PIL default resize to the native training resolution.
                gt8=np.asarray(Image.fromarray((rgba[...,:3]*rgba[...,3:4]*255).astype('uint8')).resize((1600,900)))
                target=np.asarray(Image.fromarray(rgba[...,3]).resize((1600,900),Image.Resampling.BILINEAR))
                gt=torch.tensor(gt8,device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255
                w=r.warp_at(d,g.get_xyz.detach(),(n-1)/199)
                pkg=r.render_call(r.camera(f,meta,1600),g,w)
                pred=quantize_png(pkg['render'])[None];alpha=pkg['alpha'].squeeze().cpu().numpy()
                row={'frame':n,'PSNR':float(-10*torch.log10((pred-gt).square().mean())),
                     'SSIM':float(ssim(pred,gt)),'LPIPS':float(metric(pred*2-1,gt*2-1).mean()),
                     'MS-SSIM':float(ms_ssim(pred,gt,data_range=1.,size_average=True)),
                     'observed_coverage':observed_coverage(alpha.clip(0,1),target.clip(0,1))}
                rgb=gt8.astype(np.float32)/255
                regions={'blue':(rgb[...,2]>rgb[...,0]*1.3+.02)&(rgb[...,2]>rgb[...,1]*1.05)&(target>.95),
                         'duck':(np.minimum(rgb[...,0],rgb[...,1])>rgb[...,2]*1.35+.03)&(rgb[...,0]>.15)&(rgb[...,1]>.13)&(target>.95)}
                for region,mask in regions.items():
                    mt=torch.tensor(mask,device='cuda')
                    row[region+'_pixels']=int(mask.sum())
                    row[region+'_PSNR']=float(-10*torch.log10((pred[0,:,mt]-gt[0,:,mt]).square().mean())) if mask.any() else None
                    row[region+'_alpha_below09']=float((alpha[mask]<.9).mean()) if mask.any() else None
                rows.append(row)
                if len(rows)%10==0:print(name,split,len(rows),'/',expected,flush=True)
            result={'model':name,'split':split,'count':expected,'full_state':proof,'native_resolution':[1600,900],
                    'evaluator_sha256':sha256(Path(__file__)),'GT':'original independently held-out observed camera/time RGBA',
                    'region_definition':'fixed GT color subsets, not material IDs; PSNR and alpha coverage only',
                    'metrics':{k:float(np.mean([x[k] for x in rows if x[k] is not None])) for k in keys},'frames':rows}
            write_json(path,result);print(name,split,result['metrics'],flush=True)
        del g,d;torch.cuda.empty_cache()


@torch.no_grad()
def constraints():
    from variants.transfer_integrity.runtime import Tuning
    from variants.transfer_integrity.run import config,DATA
    binding=read_json(r.REPORT/'data_binding.json')
    for name,directory in r.MODELS.items():
        g,d,proof=select(directory)
        temp=ROOT/'runs/pidg_transfer_integrity/evaluation'/name;temp.mkdir(parents=True,exist_ok=True)
        write_json(temp/'metric_config.json',config(True,binding))
        t=Tuning(temp/'metric_config.json',temp);t.transfer_deform=d;t.initialize_data(DATA)
        rows=[]
        for i in range(1,121):
            _,terms=t.geometry_loss(50000+i*10,g)
            rows.append({k:float(v) if torch.is_tensor(v) else v for k,v in terms.items()})
        write_json(r.REPORT/f'{name}_constraints.json',{'full_state':proof,'samples':120,'frames':rows,
                   'training_constraint_diagnostic_only':True,
                   'means':{k:float(np.mean([x[k] for x in rows])) for k in rows[0] if k not in ['static_sample_time_frame','track_frame']}})
        del g,d,t;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['metrics','render','all'],default='all');args=parser.parse_args()
    torch.set_num_threads(4);r.OUT.mkdir(parents=True,exist_ok=True)
    training_proof()
    if args.stage in ['metrics','all']:observations();constraints()
    if args.stage in ['render','all']:
        sys.argv=[sys.argv[0]];r.main()
    freeze(prepare_code())


if __name__=='__main__':main()
