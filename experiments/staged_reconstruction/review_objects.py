"""Independent native image/contribution review; never updates a training tensor."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
import torch
from variants.staged_reconstruction.prepare_data import REPORT,RUNS,DATA,read,write,sha
from variants.staged_reconstruction.train_static import setup,camera,PIPE
from variants.staged_reconstruction.stage2.train import image
from variants.staged_reconstruction.stage2.model import ObjectScene
from output.surface_support.core import components


def uint8(rgb):return (rgb.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)


def panel(a,label):
    p=Image.fromarray(a).resize((400,225),Image.Resampling.LANCZOS)
    result=Image.new('RGB',(400,245));result.paste(p,(0,20));ImageDraw.Draw(result).text((4,4),label,fill='white')
    return np.asarray(result)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-dir',default='S2_objects');p.add_argument('--preview-only',action='store_true');p.add_argument('--recovery',action='store_true');args=p.parse_args()
    current_report=REPORT/'S2_recovery' if args.recovery else REPORT
    renderer=setup();torch.set_num_threads(4);protocol=read(current_report/'S2_protocol.json')
    if args.recovery:
        from variants.staged_reconstruction.recovery.train import RecoveryScene
        scene=RecoveryScene(protocol)
    else:scene=ObjectScene(protocol)
    directory=RUNS/args.checkpoint_dir;checkpoint=directory/'checkpoint.pt';before=sha(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);scene.load_state_dict(state['model']);step=state['iteration']
    assert state['protocol_sha256']==sha(current_report/'S2_protocol.json')
    parent=torch.load(RUNS/'S1_static/checkpoint.pt',map_location='cpu',weights_only=False)['model']
    assert all(torch.equal(v,scene.static.state_dict()[k].cpu()) for k,v in parent.items())
    assert all(v.isfinite().all() for v in state['model'].values())
    from variants.staged_reconstruction.stage2.freeze import verify
    verify()
    if args.recovery:
        from variants.staged_reconstruction.recovery.freeze import verify as verify_recovery
        verify_recovery()
    meta=read(DATA/'transforms_train.json');byid={int(Path(f['file_path']).stem):f for f in meta['frames']}
    identity=read(REPORT/'S2_identity.json');lookup=dict(zip(identity['frame_ids'],identity['early_owner_to_component']))
    out=ROOT/'output/staged_reconstruction'/('S2_recovery' if args.recovery else 'S2_preview' if args.preview_only else 'S2_objects');out.mkdir(parents=True,exist_ok=True)
    selected=[2,27,42,43,52,86,166,188,200];rows=[];panels=[]
    colors=np.array([[1.,.2,.2],[.2,1.,.2],[1.,.8,.1],[.65,.3,1.]])
    for number in selected:
        frame=byid[number];cam=camera(frame,meta,1600);g=scene.at(frame['time']);pkg=renderer(cam,g,PIPE,scene.centers.new_zeros(3),0.,0.,0.)
        rgba=image(frame,1600);gt=(rgba[...,:3]*rgba[...,3:4]*255).round().astype(np.uint8)
        masks=components(np.round(rgba*255).astype(np.uint8),minimum_area=320)
        small=components(np.round(image(frame,400)*255).astype(np.uint8));mapping=lookup[number]
        masses=[];start=len(scene.static.get_xyz)
        for k,obj in enumerate(scene.objects):
            rgb=torch.zeros_like(g.get_xyz);rgb[start:start+len(obj.get_xyz)]=1;start+=len(obj.get_xyz)
            contribution=renderer(cam,g,PIPE,scene.centers.new_zeros(3),0.,0.,0.,override_color=rgb)['render'][0]
            masses.append(contribution.cpu().numpy())
        mass=np.stack(masses);total=mass.sum(0);overlay=np.einsum('khw,kc->hwc',mass,colors)
        shares=[]
        # Component labels must agree across scales; otherwise this frame is unknown.
        for seed,component in enumerate(mapping):
            if component<0 or masks.max()!=small.max():shares.append(None);continue
            region=masks==component;denom=total[region].sum()
            shares.append(float(mass[seed,region].sum()/denom) if denom>1e-9 else 0.)
        row={'frame':number,'qualified_early_owner_to_component':mapping,'own_group_visible_mass_share':shares,
             'total_visible_dynamic_mass':float(total.sum()),'cross_scale_component_counts':[int(small.max()+1),int(masks.max()+1)]}
        rows.append(row)
        Image.fromarray(uint8(pkg['render'])).save(out/f'train_{number:04d}_step{step:05d}.png')
        panels.append(np.concatenate([panel(gt,f'{number}: observed RGB'),panel(uint8(pkg['render']),f'S2 step{step}'),
            panel((np.clip(overlay,0,1)*255).round().astype(np.uint8),'Visible group contributions; unknown after occlusion')],axis=1))
    Image.fromarray(np.concatenate(panels,axis=0)).save(out/f'S2_review_step{step:05d}.png')
    if not args.preview_only:
        import cv2
        all_meta=read(ROOT/'data/PIDG/rubber_duck/transforms_all.json')
        all_frames=sorted(all_meta['frames'],key=lambda f:int(Path(f['file_path'].replace('\\','/')).stem))
        assert len(all_frames)==200
        summaries=[]
        for view in ['fixed86','fixed188','original']:
            writer=cv2.VideoWriter(str(out/f'{view}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(800,450));assert writer.isOpened()
            for index,observed in enumerate(all_frames):
                n=int(Path(observed['file_path'].replace('\\','/')).stem);t=(n-1)/199
                pose=byid[86 if view=='fixed86' else 188] if view!='original' else observed
                pkg=scene.render(renderer,camera(pose,meta,1600),t);rgb=uint8(pkg['render'])
                if n in [2,86,188]:Image.fromarray(rgb).save(out/f'{view}_t{n:04d}_native.png')
                writer.write(cv2.cvtColor(cv2.resize(rgb,(800,450),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR))
            writer.release();capture=cv2.VideoCapture(str(out/f'{view}.mp4'));count=0
            while capture.read()[0]:count+=1
            capture.release();assert count==200
            summaries.append({'view':view,'native_render_width':1600,'video_width':800,'decoded_frames':count,
                              'display_fps':20,'acquisition_fps_unknown':True,'novel_view_GT_available':False})
            print('OBJECT_VIDEO_REVIEW',view,count,flush=True)
        write(current_report/'S2_exports.json',{'iteration':step,'checkpoint_sha256':before,'videos':summaries})
    assert sha(checkpoint)==before
    write(current_report/('S2_preview.json' if args.preview_only else 'S2_review_numeric.json'),{'iteration':step,
        'checkpoint_sha256':before,'static_exactly_matches_S1':True,'source_freeze_passed':True,
        'optimizer_finite':all(v.isfinite().all().item() for s in state['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v)),
        'frames':rows,'manual_image_review_pending':True,'physical_ID_after_ambiguity':'unknown'})
    print('OBJECT_REVIEW_NUMERIC_COMPLETE',step,flush=True)


if __name__=='__main__':main()
