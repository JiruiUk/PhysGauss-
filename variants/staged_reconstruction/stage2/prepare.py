"""Training-only identity ledger and explicit uncertain monocular depth prior."""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from output.surface_support.core import components
from output.short_clip_diagnostic.camera_math import w2c, intrinsic
from ..prepare_data import ROOT, REPORT, RUNS, DATA, read, write, sha
from .identity import directional, reciprocal, advance


def rgba(frame, width):
    path=DATA/(frame['file_path']+'.png')
    return np.asarray(Image.open(path).convert('RGBA').resize((width,width*9//16),Image.Resampling.BILINEAR))


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-object-stage',action='store_true');args=p.parse_args()
    if not args.enable_object_stage:raise SystemExit('S2 is opt-in; pass --enable-object-stage')
    torch.set_num_threads(4)
    from ..freeze import verify
    verify()
    protocol=read(REPORT/'S2_protocol.json');meta=read(DATA/'transforms_train.json')
    flowroot=ROOT/'data/derived/flow_comparison_train_only';manifest=read(flowroot/'manifest.json')
    frames=meta['frames'];numbers=[int(Path(f['file_path']).stem) for f in frames]
    assert numbers==manifest['train_frame_ids'] and len(numbers)==120
    observations=[components(rgba(f,400)) for f in frames]
    assert observations[0].max()==3
    owner=np.arange(4,dtype=np.int16);owners=[owner.copy()];links=[];hashes={}
    for i,pair in enumerate(manifest['pairs']):
        assert (pair['source'],pair['target'])==(numbers[i],numbers[i+1])
        path=flowroot/pair['file'];actual=sha(path);assert actual==manifest['output_sha256'][pair['file']]
        hashes[pair['file']]=actual
        with np.load(path,allow_pickle=False) as data:
            f=torch.from_numpy(data['forward']);b=torch.from_numpy(data['backward'])
            # Do not consume the legacy-camera motion/depth/valid arrays.
            forward=directional(observations[i],observations[i+1],f,b)
            backward=directional(observations[i+1],observations[i],b,f)
        link=reciprocal(forward,backward);owner=advance(owner,link);owners.append(owner.copy())
        links.append({'source':numbers[i],'target':numbers[i+1],'forward':forward.tolist(),
                      'reverse':backward.tolist(),'reciprocal':link.tolist(),'early_owner_to_component':owner.tolist()})
        if (i+1)%20==0:print('OBJECT_IDENTITY',i+1,'/119',owner.tolist(),flush=True)
    identity={'training_only':True,'independent_physical_ID_GT':False,'no_occlusion_reidentification':True,
        'policy':'Adjacent visible raw FB flow, unmixed >=30px, >=25% component, reciprocal unique; lost early ID stays unknown',
        'frame_ids':numbers,'early_owner_to_component':np.asarray(owners).tolist(),'component_counts':[int(x.max())+1 for x in observations],
        'links':links,'flow_manifest_sha256':sha(flowroot/'manifest.json'),'raw_pair_sha256':hashes,
        'legacy_motion_or_depth_used':False}
    write(REPORT/'S2_identity.json',identity)
    np.savez_compressed(RUNS/'S2_observations.npz',frame_ids=numbers,components=np.stack(observations),early_owner_to_component=owners)
    # Four immutable model groups. A thin initial depth slab is a prior, not depth GT.
    first=frames[0];width=800;image=rgba(first,width);mask=components(image,minimum_area=80)
    assert mask.max()==3
    reference=np.load(RUNS/'static_metadata_initialization.npz')['xyz']
    view=w2c(first['transform_matrix'],'metadata_cv');K=intrinsic(meta['camera_angle_x'],'metadata',width,width*9//16)
    cam=reference@view[:3,:3].T+view[:3,3];base_depth=float(np.median(cam[:,2]))
    inverse=np.linalg.inv(view);generator=np.random.default_rng(protocol['seed']);receipt=[]
    for object_id in range(4):
        pixels=np.argwhere(mask==object_id)
        # Sample each observed silhouette across three explicit depth layers.
        budget=protocol['maximum_points_per_object']//3
        if len(pixels)>budget:pixels=pixels[np.sort(generator.choice(len(pixels),budget,replace=False))]
        y,x=pixels.T;uv=np.stack([x,y,np.ones_like(x)],-1);rays=uv@np.linalg.inv(K).T
        depths=base_depth+np.array([-.15,0.,.15]);camera_points=(rays[None]*depths[:,None,None]).reshape(-1,3)
        world=camera_points@inverse[:3,:3].T+inverse[:3,3];center=world.mean(0);local=(world-center).astype(np.float32)
        rgb=np.tile(image[y,x,:3].astype(np.float32)/255,(3,1));spacing=base_depth/K[0,0]*1.8
        path=RUNS/f'S2_object_{object_id}.npz'
        np.savez_compressed(path,xyz=local,rgb=rgb,spacing=np.asarray(spacing),center=center.astype(np.float32),object_id=object_id)
        receipt.append({'object_id':object_id,'points':len(local),'center_prior':center.tolist(),'file':path.name,'sha256':sha(path)})
    panels=[];colors=np.array([[255,100,100],[100,255,100],[255,200,50],[180,100,255]])
    for n in [2,27,42,43,86,188,200]:
        if n not in numbers:continue
        i=numbers.index(n);im=rgba(frames[i],400).copy();comp=observations[i]
        for seed,k in enumerate(owners[i]):
            if k>=0:im[comp==k,:3]=(im[comp==k,:3]*.4+colors[seed]*.6).astype(np.uint8)
        panel=Image.fromarray(im).convert('RGB');ImageDraw.Draw(panel).text((4,4),f'{n}: early ID links {owners[i].tolist()}',fill='white');panels.append(np.asarray(panel))
    Image.fromarray(np.concatenate(panels,axis=0)).save(REPORT/'S2_identity_review.png')
    write(REPORT/'S2_data_receipt.json',{'protocol_sha256':sha(REPORT/'S2_protocol.json'),
        'objects':receipt,'base_depth_prior':base_depth,'depth_offsets':[-.15,0.,.15],
        'initialization':'First training frame silhouette backprojection into an explicit shared camera-depth slab; uncertain geometry, not GT',
        'source_frame':2,'identity_sha256':sha(REPORT/'S2_identity.json'),'observation_npz_sha256':sha(RUNS/'S2_observations.npz'),
        'train_metadata_sha256':sha(DATA/'transforms_train.json'),
        'training_image_sha256':{f['file_path']:sha(DATA/(f['file_path']+'.png')) for f in frames},
        'camera_contract':'metadata_cv','test_RGB_used':False})
    print('OBJECT_PREPARATION_COMPLETE',[(r['object_id'],r['points']) for r in receipt],flush=True)


if __name__=='__main__':main()
