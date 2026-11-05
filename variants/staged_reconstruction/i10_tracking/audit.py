"""Independent process replay of all train-only tracker and geometry decisions."""
import argparse
import sys
from collections import Counter
import numpy as np
from PIL import Image
import torch
from .io import BASE,REPORT,DATA,SOURCE,WEIGHTS,read,write,verify
from .core import seeds,grayscale,qualify


@torch.no_grad()
def run():
    verify();torch.set_num_threads(4);p=read(BASE/'protocol.json')
    sys.path.insert(0,str(SOURCE))
    from cotracker.predictor import CoTrackerPredictor
    model=CoTrackerPredictor(checkpoint=str(WEIGHTS),offline=True,v2=False,window_len=60).cuda().eval()
    meta=read(DATA/'transforms_train.json');rgbs=[];grays=[]
    for f in meta['frames']:
        a=np.asarray(Image.open(DATA/(f['file_path']+'.png')).convert('RGBA').resize((400,225),Image.Resampling.BILINEAR))
        rgb=np.uint8(a[...,:3].astype(np.float32)*a[...,3:4]/255);rgbs.append(rgb);grays.append(grayscale(rgb))
    totals=Counter();max_error=0
    for start in p['starts']:
        saved=np.load(REPORT/f'clip_{start:03d}.npz');points=seeds(grays[start],p)
        assert np.array_equal(points,saved['seeds'])
        video=torch.from_numpy(np.stack(rgbs[start:start+12])).permute(0,3,1,2)[None].float().cuda()
        query=torch.tensor(np.c_[np.zeros(len(points)),points],device='cuda',dtype=torch.float32)[None]
        f,v=model(video,queries=query,backward_tracking=False)
        rq=torch.cat([query[...,:1],f[:,-1]],-1)
        r,rv=model(video.flip(1).contiguous(),queries=rq,backward_tracking=False)
        actual=dict(forward=f[0].cpu().numpy(),reverse=r[0].flip(0).cpu().numpy(),
                    visible=v[0].cpu().numpy(),reverse_visible=rv[0].flip(0).cpu().numpy())
        for key,value in actual.items():
            assert np.array_equal(value,saved[key]),(start,key)
        good,diagnostic=qualify(actual['forward'],actual['reverse'],actual['visible'],actual['reverse_visible'],grays[start:start+12],p)
        assert np.array_equal(good,saved['accepted'])
        assert diagnostic==read(REPORT/f'clip_{start:03d}.json')['diagnostic']
        totals['seeds']+=len(points);totals['consistent']+=int(good.sum())
    assert totals['consistent']==len(read(REPORT/'tracks.json')['tracks'])
    verify();write(REPORT/'tracker_replay.json',dict(passed=True,windows=28,all_raw_forward_reverse_visibility_seeds_and_gates_exact=True,**dict(totals)))
    print('MASK_FREE_TRACKER_INDEPENDENT_REPLAY',dict(totals),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--enable-replay',action='store_true');args=parser.parse_args()
    if not args.enable_replay:raise SystemExit('Audit is opt-in')
    run()
