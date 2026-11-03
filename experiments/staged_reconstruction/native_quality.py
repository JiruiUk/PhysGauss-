"""Standard native RGB SSIM/VGG-LPIPS; read-only, no selection or metric tuning."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from variants.staged_reconstruction.prepare_data import REPORT,RUNS,read,write,sha
from variants.staged_reconstruction.train_static import setup,camera
from variants.staged_reconstruction.stage2.train import image
from variants.staged_reconstruction.stage4.bind import load_physics_state


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--stages',nargs='+',choices=['S2','S3','S4','S5','S6'],required=True);a=p.parse_args()
    renderer=setup();torch.set_num_threads(4)
    from utils.loss_utils import ssim
    import lpips
    vgg=Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
    linear=Path(lpips.__file__).parent/'weights/v0.1/vgg.pth'
    assert vgg.exists() and linear.exists(),'Use the established local standard metric weights; no network download or replacement'
    metric=lpips.LPIPS(net='vgg',version='0.1').cuda().eval().requires_grad_(False)
    contract={'native_size':[1600,900],'image_protocol':'RGB clipped[0,1] and rounded8bit; original RGBA composited on black',
        'LPIPS':'Standard lpips package, VGG v0.1 calibrated weights, inputs[-1,1], not task-trained',
        'VGG_sha256':sha(vgg),'LPIPS_linear_sha256':sha(linear),'metric_code_sha256':sha(Path(__file__)),
        'scope':'Full-frame observed-camera image quality; background can dominate; not independent geometry or physical GT',
        'legacy_PIDG_comparison':'Different camera/capacity branch; these are sequential checkpoints, not matched baseline ablations'}
    for stage in a.stages:
        report=REPORT/('S2_geometry_recovery' if stage=='S2' else stage)
        path=RUNS/('S2_geometry_recovery/S2_objects' if stage=='S2' else stage+'/formal')/'checkpoint.pt';before=sha(path)
        result_path=report/'native_full_quality.json'
        if result_path.exists():
            previous=read(result_path)
            assert previous['checkpoint_sha256']==before and previous['contract']==contract
            print('QUALITY_ALREADY_VERIFIED',stage,flush=True);continue
        if stage=='S2':
            from variants.staged_reconstruction.geometry_recovery.model import BoundedScene
            scene=BoundedScene(read(report/'S2_protocol.json'))
            scene.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
        else:scene,_,_=load_physics_state(path)
        splits={}
        for split in ['val','test']:
            meta=read(ROOT/f'data/derived/kinematic_resplit/transforms_{split}.json');rows=[]
            for frame in meta['frames']:
                rgba=image(frame,1600);gt=torch.tensor(rgba[...,:3]*rgba[...,3:4],device='cuda').permute(2,0,1)[None]
                raw=scene.render(renderer,camera(frame,meta,1600),frame['time'])['render'].clamp(0,1)[None]
                pred=(raw*255).round()/255
                rows.append({'frame':int(Path(frame['file_path']).stem),
                    'PSNR':float(-10*torch.log10((pred-gt).square().mean().clamp_min(1e-12))),
                    'SSIM':float(ssim(pred,gt)),'LPIPS':float(metric(pred*2-1,gt*2-1).mean())})
            splits[split]={'count':len(rows),'frames':rows,'metrics':{k:float(np.mean([x[k] for x in rows])) for k in ['PSNR','SSIM','LPIPS']}}
            print('NATIVE_STANDARD_QUALITY',stage,split,splits[split]['metrics'],flush=True)
        assert sha(path)==before
        write(result_path,{'stage':stage,'checkpoint_sha256':before,'contract':contract,'splits':splits})
        del scene;torch.cuda.empty_cache()


if __name__=='__main__':main()
