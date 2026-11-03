"""Independent fixed-pair conditioning diagnostic; reads training cues only."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from variants.staged_reconstruction.prepare_data import ROOT, DATA, read, write, sha
from variants.staged_reconstruction.train_static import camera, PIPE
from variants.staged_reconstruction.motion_integrity.prepare import load_parent, PARENT
from variants.staged_reconstruction.ordinal_depth.prepare import EXP, WORK


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--arm',choices=['parent','I3'],required=True); a=p.parse_args()
    torch.set_num_threads(4)
    renderer,scene,_,_,_=load_parent()
    path=PARENT if a.arm=='parent' else WORK/'formal/checkpoint.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    if a.arm=='I3': assert state['iteration']==2000
    scene.load_state_dict(state['model'])
    meta=read(DATA/'transforms_train.json'); plan=read(EXP/'protocol.json')
    frames={int(Path(f['file_path']).stem):f for f in meta['frames']}
    rows=[]
    for number in [2,43,86,146,188,200]:
        f=frames[number]; cam=camera(f,meta,800)
        package=renderer(cam,scene.dynamic_at(f['time']),PIPE,scene.centers.new_zeros(3),0.,0.,0.)
        z=(package['depth'].squeeze()/package['alpha'].squeeze().clamp_min(1e-6)).flatten()
        with np.load(WORK/f'pairs/{number:04d}.npz') as data:
            pairs=torch.tensor(data['pairs'],device='cuda'); signs=torch.tensor(data['signs'],device='cuda')
        difference=z[pairs[:,0]]-z[pairs[:,1]]
        response=torch.tanh(plan['ordinal_sharpness']*difference)
        wrong=difference*signs<0
        derivative=plan['ordinal_sharpness']*(1-response.square())
        rows.append({'frame':number,'pairs':len(pairs),'wrong_order_fraction':float(wrong.float().mean()),
                     'wrong_pairs_near_zero_tanh_gradient_fraction':float((derivative[wrong]<1e-5).float().mean()),
                     'loss':float((response-signs).abs().mean()),
                     'camera_depth_difference_quantiles':torch.quantile(difference.abs(),difference.new_tensor([.1,.5,.9,.99])).cpu().tolist()})
    out=EXP/f'conditioning_{a.arm}.json'; assert not out.exists()
    write(out,{'checkpoint_sha256':sha(path),'rows':rows,'test_RGB_read':False,'diagnostic_not_GT':True})
    print(rows,flush=True)


if __name__=='__main__': main()
