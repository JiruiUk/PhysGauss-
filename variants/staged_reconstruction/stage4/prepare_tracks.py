"""No training/row binding. Output provenance records estimated image evidence."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
import torch
from ..prepare_data import ROOT,REPORT,RUNS,DATA,read,write,sha
from .tracks import advect


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-track-preparation',action='store_true');args=p.parse_args()
    if not args.enable_track_preparation:raise SystemExit('Sparse track preparation is opt-in')
    torch.set_num_threads(4);report=REPORT/'S4';report.mkdir(parents=True,exist_ok=True)
    ledger=np.load(RUNS/'S2_observations.npz',allow_pickle=False);ids=ledger['frame_ids'];components=ledger['components'];owner=ledger['early_owner_to_component']
    manifest=read(ROOT/'data/derived/flow_comparison_train_only/manifest.json');assert list(ids)==manifest['train_frame_ids']
    images=[];image_hashes={};flows=[];flow_hashes={}
    for n in ids:
        path=DATA/f'train/{n:04d}.png';image_hashes[str(int(n))]=sha(path)
        rgb=np.asarray(Image.open(path).convert('RGB').resize((400,225),Image.Resampling.BILINEAR))
        images.append(rgb)
    for pair in manifest['pairs']:
        path=ROOT/'data/derived/flow_comparison_train_only'/pair['file'];assert sha(path)==manifest['output_sha256'][pair['file']]
        with np.load(path,allow_pickle=False) as data:flows.append((torch.tensor(data['forward']),torch.tensor(data['backward'])))
        flow_hashes[pair['file']]=sha(path)
    tracks=[];source_image_patches=[]
    for start in range(0,len(ids)-2,5):
        for k in range(4):
            comp=int(owner[start,k])
            if comp<0:continue
            mask=binary_erosion(components[start]==comp,iterations=2).astype(np.uint8)*255
            corners=cv2.goodFeaturesToTrack(cv2.cvtColor(images[start],cv2.COLOR_RGB2GRAY),maxCorners=24,qualityLevel=.03,minDistance=3,mask=mask,blockSize=3)
            if corners is None:continue
            points=torch.tensor(corners[:,0,:]);initial=points.clone();alive=torch.ones(len(points),dtype=torch.bool)
            history=[points.numpy().copy()];observed=[int(ids[start])]
            for target in range(start+1,min(start+5,len(ids))):
                if owner[target,k]<0 or owner[target-1,k]<0:break
                f,b=flows[target-1]
                points,good=advect(points,f,b,torch.tensor(components[target-1]),torch.tensor(components[target]),int(owner[target-1,k]),int(owner[target,k]))
                alive &=good;history.append(points.numpy().copy());observed.append(int(ids[target]))
            if len(history)<3:continue
            # Reverse the full fragment independently and reject accumulated drift.
            returned=points.clone()
            for previous in range(start+len(history)-2,start-1,-1):
                f,b=flows[previous]
                returned,good=advect(returned,b,f,torch.tensor(components[previous+1]),torch.tensor(components[previous]),int(owner[previous+1,k]),int(owner[previous,k]))
                alive &=good
            closure=(returned-initial).norm(dim=-1);alive &=closure<=1.
            for index in torch.nonzero(alive).flatten().tolist():
                tracks.append({'track_id':len(tracks),'local_early_object':k,'source_frame':observed[0],
                    'frames':observed,'uv_400':np.asarray(history)[:,index].tolist(),'reverse_closure_px':float(closure[index]),
                    'physical_material_ID_GT':False,'gaussian_row':None})
        if start%20==0:print('SPARSE_IMAGE_FRAGMENTS',start,'/',len(ids),'tracks',len(tracks),flush=True)
    result={'training_only':True,'image_space':[400,225],'frames':ids.tolist(),'tracks':tracks,
        'input_image_sha256':image_hashes,'raw_flow_pair_sha256':flow_hashes,'flow_manifest_sha256':sha(ROOT/'data/derived/flow_comparison_train_only/manifest.json'),
        'policy':'Eroded yellow Shi-Tomasi corners,3-5 observations, raw bidirectional flow<=0.75px, full fragment reverse closure<=1px; qualified early identity only',
        'not_GT':'Estimated image correspondences can follow reflections or shared flow bias; no dense 3D/material truth or occlusion re-ID',
        'old_camera_motion_or_depth_used':False,'heldout_images_used':False,'gaussian_bindings_pending_admitted_S3':True}
    write(report/'image_tracks.json',result)
    print('SPARSE_TRACK_PREPARATION_COMPLETE',len(tracks),flush=True)


if __name__=='__main__':main()
