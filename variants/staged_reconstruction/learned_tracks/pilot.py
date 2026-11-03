"""Isolated, opt-in CoTracker3 train-only pilot. No PIDG imports or updates."""
import argparse
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from ..prepare_data import ROOT, DATA, read, write, sha
from ..surface_trajectory.observations import appearance_masks
from .fetch import EXP, SOURCE, WEIGHTS, main as verify_assets
from .quality import qualify

CONFIG = dict(version=1, starts=[0,54,108], clip_length=12, width=400,
              max_corners=64, quality_level=.01, seed_distance=4., mask_erosion=2,
              min_observations=8, patch_radius=2, patch_std_min=2., ncc_min=.6,
              closure_max_px=.75, direction_max_px=.75,
              no_occlusion_reidentification=True, model_confidence_is_not_identity=True)


def montage(images, tracks, accepted, ids, target):
    canvas=Image.new('RGB',(4*400,3*247),'#111111')
    for t, rgb in enumerate(images):
        tile=Image.fromarray(rgb.copy()); draw=ImageDraw.Draw(tile)
        for j, good in enumerate(accepted):
            color=(0,255,0) if good else (255,80,80)
            point=tracks[t,j]
            if not np.isfinite(point).all(): continue
            x,y=map(float,point); draw.ellipse((x-2,y-2,x+2,y+2),outline=color)
            if t>0 and np.isfinite(tracks[:t+1,j]).all():
                draw.line([tuple(map(float,p)) for p in tracks[max(0,t-2):t+1,j]],fill=color,width=1)
        x,y=(t%4)*400,(t//4)*247
        canvas.paste(tile,(x,y+22))
        ImageDraw.Draw(canvas).text((x+4,y+4),f'train frame {ids[t]} | green=qualified red=rejected',fill='white')
    canvas.save(target)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--enable-learned-track-pilot',action='store_true')
    args=parser.parse_args()
    if not args.enable_learned_track_pilot: raise SystemExit('Learned-track pilot is opt-in')
    output=EXP/'pilot_v1'; output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists(): raise SystemExit('Preserve completed pilot')
    if (output/'protocol.json').exists(): assert read(output/'protocol.json')==CONFIG
    else: write(output/'protocol.json',CONFIG)
    verify_assets()
    torch.set_num_threads(4)
    free,_=torch.cuda.mem_get_info()
    assert free>6*1024**3, 'Defer inference until >6GiB is free; do not disturb other jobs'
    # Official imports reset torch RNG: this process is intentionally isolated.
    sys.path.insert(0,str(SOURCE))
    from cotracker.predictor import CoTrackerPredictor
    tracker=CoTrackerPredictor(checkpoint=WEIGHTS.as_posix(),offline=True,v2=False,window_len=60).cuda().eval()
    metadata=DATA/'transforms_train.json'
    frames=sorted(read(metadata)['frames'],key=lambda f:f['time'])
    assert len(frames)==120 and len({f['time'] for f in frames})==120
    windows=[]; image_hashes={}; began=time.monotonic()
    for start in CONFIG['starts']:
        clip=frames[start:start+CONFIG['clip_length']]
        rgbs, masks, grays, ids=[],[],[],[]
        for frame in clip:
            path=DATA/(frame['file_path']+'.png'); image_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
            rgba=np.asarray(Image.open(path).convert('RGBA').resize((400,225),Image.Resampling.BILINEAR))
            rgbs.append(np.uint8(rgba[...,:3].astype(np.float32)*rgba[...,3:4]/255))
            _,mask=appearance_masks(rgba.astype(np.float32)/255,CONFIG['mask_erosion'],True)
            masks.append(mask.astype(np.float32))
            grays.append(cv2.cvtColor(rgbs[-1],cv2.COLOR_RGB2GRAY).astype(np.float32))
            ids.append(int(frame['file_path'].split('/')[-1]))
        corners=cv2.goodFeaturesToTrack(grays[0],maxCorners=CONFIG['max_corners'],qualityLevel=CONFIG['quality_level'],
                    minDistance=CONFIG['seed_distance'],mask=(masks[0]*255).astype(np.uint8),blockSize=3)
        assert corners is not None
        seeds=corners[:,0]
        video=torch.from_numpy(np.stack(rgbs)).permute(0,3,1,2)[None].float().cuda()
        queries=torch.tensor(np.c_[np.zeros(len(seeds)),seeds],device='cuda',dtype=torch.float32)[None]
        forward,visible=tracker(video,queries=queries,backward_tracking=False)
        reverse_queries=torch.cat([queries[...,:1],forward[:,-1]],dim=-1)
        reverse,reverse_visible=tracker(video.flip(1).contiguous(),queries=reverse_queries,backward_tracking=False)
        f=forward[0].cpu().numpy(); r=reverse[0].flip(0).cpu().numpy()
        v=visible[0].cpu().numpy(); rv=reverse_visible[0].flip(0).cpu().numpy()
        accepted,diagnostics=qualify(f,r,v,rv,masks,grays,CONFIG)
        np.savez_compressed(output/f'clip_{start:03d}.npz',forward=f,reverse=r,visible=v,reverse_visible=rv,accepted=accepted)
        montage(rgbs,f,accepted,ids,output/f'clip_{start:03d}.png')
        row={'start_index':start,'frames':ids,'times':[frame['time'] for frame in clip],
             'seeds':len(seeds),'qualified':int(accepted.sum()),'accepted_rows':np.flatnonzero(accepted).tolist(),
             'diagnostics':diagnostics,'actual_visual_review':False}
        windows.append(row); write(output/f'clip_{start:03d}.json',row)
        print('LEARNED_TRACK_PILOT',start,len(seeds),int(accepted.sum()),'elapsed',round(time.monotonic()-began,1),flush=True)
        del video,queries,forward,visible,reverse,reverse_visible,reverse_queries
        torch.cuda.empty_cache()
    write(output/'result.json',{'windows':windows,'training_only':True,'metadata_sha256':sha(metadata),
             'input_sha256':image_hashes,'source_lock_sha256':sha(EXP/'source_lock.json'),
             'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in list(Path(__file__).parent.glob('*.py'))+[ROOT/'tests/test_learned_track_quality.py']},
             'protocol_sha256':sha(output/'protocol.json'),'optimizer_updates':0,
             'heldout_RGB_used':False,'material_identity_GT':False,'actual_visual_review':False,
             'training_admitted':False,'maximum_GPU_memory_bytes':torch.cuda.max_memory_allocated(),
             'forced_query_predictions_excluded_from_validation':True})


if __name__=='__main__': main()
