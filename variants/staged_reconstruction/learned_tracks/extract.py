"""Full train-only learned image tracks. Evidence binding remains separate."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image
import torch
from ..prepare_data import ROOT, DATA, read, write, sha
from ..surface_trajectory.observations import appearance_masks
from .fetch import EXP,SOURCE,WEIGHTS,main as verify_assets
from .pilot import CONFIG as PILOT_CONFIG,montage
from .quality import qualify
from .seeds import proposals

CONFIG=dict(PILOT_CONFIG,version=2,starts=list(range(0,109,4)),seed_policy='dense textured5x5 patches, greedy4px spacing',
            acceptance_thresholds_unchanged=True,material_identity_GT=False)
OUT=EXP/'observations_v1'


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--enable-learned-track-extraction',action='store_true')
    args=parser.parse_args()
    if not args.enable_learned_track_extraction: raise SystemExit('Learned-track extraction is opt-in')
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'image_tracks.json').exists(),'Preserve completed observations'
    if (OUT/'protocol.json').exists(): assert read(OUT/'protocol.json')==CONFIG
    else: write(OUT/'protocol.json',CONFIG)
    verify_assets(); torch.set_num_threads(4)
    free,_=torch.cuda.mem_get_info(); assert free>6*1024**3,'Defer until >6GiB free'
    sys.path.insert(0,str(SOURCE))
    from cotracker.predictor import CoTrackerPredictor
    model=CoTrackerPredictor(checkpoint=WEIGHTS.as_posix(),offline=True,v2=False,window_len=60).cuda().eval()
    meta_path=DATA/'transforms_train.json'
    frames=sorted(read(meta_path)['frames'],key=lambda f:f['time'])
    assert len(frames)==120
    rgbs,masks,grays,hashes,ids=[],[],[],{},[]
    for frame in frames:
        path=DATA/(frame['file_path']+'.png'); hashes[path.relative_to(ROOT).as_posix()]=sha(path)
        rgba=np.asarray(Image.open(path).convert('RGBA').resize((400,225),Image.Resampling.BILINEAR))
        rgb=np.uint8(rgba[...,:3].astype(np.float32)*rgba[...,3:4]/255)
        rgbs.append(rgb); grays.append(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32))
        masks.append(appearance_masks(rgba.astype(np.float32)/255,2,True)[1].astype(np.float32))
        ids.append(int(Path(frame['file_path']).stem))
    tracks,windows=[],[]; began=time.monotonic()
    for start in CONFIG['starts']:
        end=start+CONFIG['clip_length']; key=f'clip_{start:03d}'
        if (OUT/f'{key}.json').exists():
            window=read(OUT/f'{key}.json'); windows.append(window); tracks.extend(window['tracks']); continue
        seeds=proposals(grays[start],masks[start],CONFIG['max_corners'],CONFIG['seed_distance'],CONFIG['patch_std_min'])
        assert len(seeds)
        video=torch.from_numpy(np.stack(rgbs[start:end])).permute(0,3,1,2)[None].float().cuda()
        query=torch.tensor(np.c_[np.zeros(len(seeds)),seeds],device='cuda',dtype=torch.float32)[None]
        f,v=model(video,queries=query,backward_tracking=False)
        rq=torch.cat([query[...,:1],f[:,-1]],-1)
        r,rv=model(video.flip(1).contiguous(),queries=rq,backward_tracking=False)
        f0=f[0].cpu().numpy(); r0=r[0].flip(0).cpu().numpy()
        v0=v[0].cpu().numpy(); rv0=rv[0].flip(0).cpu().numpy()
        accepted,diagnostic=qualify(f0,r0,v0,rv0,masks[start:end],grays[start:end],CONFIG)
        accepted_tracks=[]
        for row in np.flatnonzero(accepted):
            accepted_tracks.append({'source_index':start,'source_frame':ids[start],
                'frames':ids[start:end],'times':[frame['time'] for frame in frames[start:end]],
                'uv_400':f0[:,row].tolist(),'confidence':[float(np.exp(-diagnostic['closure_px'][row]))]*CONFIG['clip_length'],
                'reverse_closure_px':diagnostic['closure_px'][row],'bootstrap_count':3,
                'time_band':min(2,int(frames[start]['time']*3)),'physical_material_ID_GT':False,
                'occlusion_reidentified':False,'source_row':int(row)})
        montage(rgbs[start:end],f0,accepted,ids[start:end],OUT/f'{key}.png')
        np.savez_compressed(OUT/f'{key}.npz',forward=f0,reverse=r0,visible=v0,reverse_visible=rv0,accepted=accepted)
        window={'start':start,'seeds':len(seeds),'qualified':int(accepted.sum()),'tracks':accepted_tracks,'diagnostic':diagnostic}
        write(OUT/f'{key}.json',window); windows.append(window); tracks.extend(accepted_tracks)
        print('LEARNED_TRACK_EXTRACT',start,len(seeds),int(accepted.sum()),'elapsed',round(time.monotonic()-began,1),flush=True)
        del video,query,f,r,v,rv,rq; torch.cuda.empty_cache()
    for index,track in enumerate(tracks): track['track_id']=index
    result={'stage':'R0_learned','tracks':tracks,'windows':[{'start':w['start'],'seeds':w['seeds'],'qualified':w['qualified']} for w in windows],
            'counts_by_time_band':dict(Counter(t['time_band'] for t in tracks)),
            'training_only':True,'image_space':[400,225],'image_sha256':hashes,'metadata_sha256':sha(meta_path),
            'source_lock_sha256':sha(EXP/'source_lock.json'),'protocol_sha256':sha(OUT/'protocol.json'),
            'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in list(Path(__file__).parent.glob('*.py'))+list((ROOT/'tests').glob('test_learned_track_*.py'))},
            'maximum_GPU_memory_bytes':torch.cuda.max_memory_allocated(),'optimizer_updates':0,'heldout_RGB_used':False,
            'human_review_completed':False,'binding_pending':True,'material_identity_GT':False,
            'no_object_count_or_material_segmentation':True,'query_overwrites_excluded':True}
    write(OUT/'image_tracks.json',result)
    print('LEARNED_TRACK_EXTRACTION_COMPLETE',len(tracks),result['counts_by_time_band'],flush=True)


if __name__=='__main__': main()
