"""Train-only, full-image, bidirectional CoTracker3 inference; no PIDG mutation."""
import argparse
from collections import Counter
import sys
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
from .io import ROOT,BASE,REPORT,RUNS,OUT,DATA,SOURCE,WEIGHTS,read,write,sha,lock,verify,stopped
from .core import grayscale,seeds,qualify


def montage(images,tracks,accepted,ids,path):
    canvas=Image.new('RGB',(1600,741),'#111111')
    for t,rgb in enumerate(images):
        tile=Image.fromarray(rgb);draw=ImageDraw.Draw(tile)
        for j,good in enumerate(accepted):
            if not np.isfinite(tracks[t,j]).all():continue
            x,y=map(float,tracks[t,j]);color=(30,255,80) if good else (255,90,90)
            draw.ellipse((x-2,y-2,x+2,y+2),outline=color)
            if t and np.isfinite(tracks[max(t-2,0):t+1,j]).all():
                draw.line([tuple(map(float,p)) for p in tracks[max(t-2,0):t+1,j]],fill=color,width=1)
        x,y=(t%4)*400,(t//4)*247;canvas.paste(tile,(x,y+22))
        ImageDraw.Draw(canvas).text((x+4,y+4),f'train {ids[t]} | green=consistent red=unknown',fill='white')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


@torch.no_grad()
def run():
    with lock():
        verify();p=read(BASE/'protocol.json');torch.set_num_threads(4)
        assert not (REPORT/'tracks.json').exists(),'Preserve completed extraction'
        assert torch.cuda.mem_get_info()[0]>6*1024**3,'Wait for GPU; do not stop other tasks'
        sys.path.insert(0,str(SOURCE))
        from cotracker.predictor import CoTrackerPredictor
        tracker=CoTrackerPredictor(checkpoint=str(WEIGHTS),offline=True,v2=False,window_len=60).cuda().eval()
        meta=read(DATA/'transforms_train.json');frames=meta['frames']
        assert len(frames)==120 and all(a['time']<b['time'] for a,b in zip(frames,frames[1:]))
        images=[];grays=[];ids=[]
        for frame in frames:
            rgba=np.asarray(Image.open(DATA/(frame['file_path']+'.png')).convert('RGBA').resize((400,225),Image.Resampling.BILINEAR))
            rgb=np.uint8(rgba[...,:3].astype(np.float32)*rgba[...,3:4]/255)
            images.append(rgb);grays.append(grayscale(rgb));ids.append(int(frame['file_path'].split('/')[-1]))
        tracks=[];windows=[];began=time.monotonic()
        for start in p['starts']:
            assert not stopped(),'Stopped before next window; completed arrays remain saved'
            file=REPORT/f'clip_{start:03d}.json'
            if file.exists():
                window=read(file);windows.append(window);tracks+=window['tracks'];continue
            points=seeds(grays[start],p);assert len(points)>0
            video=torch.from_numpy(np.stack(images[start:start+12])).permute(0,3,1,2)[None].float().cuda()
            query=torch.tensor(np.c_[np.zeros(len(points)),points],device='cuda',dtype=torch.float32)[None]
            f,v=tracker(video,queries=query,backward_tracking=False)
            rq=torch.cat([query[...,:1],f[:,-1]],-1)
            r,rv=tracker(video.flip(1).contiguous(),queries=rq,backward_tracking=False)
            f=f[0].cpu().numpy();r=r[0].flip(0).cpu().numpy();v=v[0].cpu().numpy();rv=rv[0].flip(0).cpu().numpy()
            good,diagnostics=qualify(f,r,v,rv,grays[start:start+12],p)
            rows=[dict(id=f'{start:03d}_{row:03d}',start=start,source_row=int(row),
                       uv=f[:,row].tolist(),times=[a['time'] for a in frames[start:start+12]],
                       band=min(2,start//40)) for row in np.flatnonzero(good)]
            np.savez_compressed(REPORT/f'clip_{start:03d}.npz',seeds=points,forward=f,reverse=r,visible=v,reverse_visible=rv,accepted=good)
            montage(images[start:start+12],f,good,ids[start:start+12],OUT/f'clip_{start:03d}.png')
            window=dict(start=start,seeds=len(points),qualified=int(good.sum()),tracks=rows,diagnostic=diagnostics)
            write(file,window);windows.append(window);tracks+=rows
            print('MASK_FREE_TRACKS',start,len(points),int(good.sum()),round(time.monotonic()-began,2),flush=True)
            del video,query,rq;torch.cuda.empty_cache()
        verify()
        write(REPORT/'tracks.json',dict(tracks=tracks,windows=[{k:w[k] for k in ['start','seeds','qualified']} for w in windows],
            total_seeds=sum(w['seeds'] for w in windows),consistent_tracks=len(tracks),
            consistent_by_band=dict(Counter(t['band'] for t in tracks)),optimizer_updates=0,
            color_masks_or_static_reference=False,unknowns_preserved=True,train_only=True,
            freeze_sha256=sha(REPORT/'source_freeze.json'),duration_seconds=time.monotonic()-began))
        write(RUNS/'state.json',dict(status='tracks_complete_binding_pending',optimizer_updates=0,pending=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-mask-free-tracks',action='store_true');args=parser.parse_args()
    if not args.enable_mask_free_tracks:raise SystemExit('Mask-free tracking is opt-in')
    run()

if __name__=='__main__':main()
