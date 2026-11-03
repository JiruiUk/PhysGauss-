"""Isolated pretrained depth inference; never imports a PIDG renderer or trainer."""
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
import torch
from ..prepare_data import ROOT,DATA,read,write,sha
from .fetch import EXP,SOURCE,WEIGHTS
from .core import windows,GATES


def main():
    assert not (EXP/'sequence_inference.json').exists(),'Preserve pilot versions'
    lock=read(EXP/'source_lock.json')
    assert sha(WEIGHTS)==lock['model_sha256']
    assert all(sha(SOURCE/name)==value for name,value in lock['code_sha256'].items())
    meta=read(DATA/'transforms_train.json'); frames={int(Path(f['file_path']).stem):f for f in meta['frames']}
    clips={0:sorted(frames)}; input_size=280
    write(EXP/'sequence_protocol.json',{'anchors':[2,43,86,146,188,200],'windows':clips,'input_size':input_size,
        'quality_gates':GATES,'RGB_source':'original training120 only','test_or_val_RGB_used':False,
        'model':'official Small Metric','global_static_scale_only':True,
        'irregular_training_time_spacing':True,'no_missing_RGB_interpolation':True,
        'official_inference_replication_padding_retained':True,'optimizer_updates':0})
    sys.path.insert(0,str(ROOT/'third_party/video_depth_vendor'))
    sys.path.insert(0,str(SOURCE))
    from video_depth_anything.video_depth import VideoDepthAnything
    torch.set_num_threads(4)
    model=VideoDepthAnything(encoder='vits',features=64,out_channels=[48,96,192,384],metric=True)
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    assert all(value.isfinite().all() for value in state.values())
    model.load_state_dict(state,strict=True); model=model.cuda().eval().requires_grad_(False)
    output=ROOT/'runs/pidg_staged_reconstruction/video_depth/sequence'; output.mkdir(parents=True,exist_ok=True)
    receipts={}
    for anchor,ids in clips.items():
        rgb=[]; hashes={}
        for number in ids:
            path=DATA/(frames[number]['file_path']+'.png'); hashes[str(number)]=sha(path)
            with Image.open(path) as image:
                rgba=np.array(image.convert('RGBA').resize((800,450),Image.Resampling.BILINEAR))
            rgb.append((rgba[...,:3].astype(np.float32)*(rgba[...,3:4]/255.)).round().astype(np.uint8))
        torch.cuda.reset_peak_memory_stats()
        depth,_=model.infer_video_depth(np.stack(rgb),30,input_size=input_size,device='cuda',fp32=False)
        assert depth.shape==(len(ids),450,800) and np.isfinite(depth).all()
        path=output/f'anchor{anchor:04d}.npz'; assert not path.exists()
        np.savez_compressed(path,frame_ids=ids,depth=depth)
        receipts[str(anchor)]={'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),
            'source_RGB_hashes':hashes,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
        print('VIDEO_DEPTH_FULL_TRAIN_SEQUENCE',anchor,ids,depth.min(),depth.max(),flush=True)
    write(EXP/'sequence_inference.json',{'receipts':receipts,'source_commit':lock['commit'],
        'model_sha256':sha(WEIGHTS),'actual_repository_inference_executed':True,
        'optimizer_updates':0,'PIDG_training_changed':False,'user_data_uploaded':False})


if __name__=='__main__': main()
