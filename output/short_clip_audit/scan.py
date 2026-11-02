"""Training-data-only short-clip search; never loads C1 or edits datasets."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from scipy import ndimage
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
REPORT=ROOT/'experiments/short_clip_audit';OUT=ROOT/'output/short_clip_audit'
TRAIN=ROOT/'data/derived/kinematic_resplit/training';FLOW=ROOT/'data/derived/flow_comparison_train_only'
from output.surface_support.core import propagate
from variants.surface_correspondence.model import qualify_fragments
from variants.flow_comparison.geometry import forward_backward_valid
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')


def rgba(frame,width=400):
    with Image.open(TRAIN/(frame['file_path']+'.png')) as im:return np.array(im.convert('RGBA').resize((width,width*9//16),Image.Resampling.BILINEAR))


def rotation(matrix):
    u,_,v=np.linalg.svd(matrix[:3,:3]);q=u@v
    if np.linalg.det(q)<0:u[:,-1]*=-1;q=u@v
    return q


def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    parent=read(ROOT/'experiments/surface_correspondence/input_lock.json')
    own=[Path(__file__),REPORT/'plan.json',ROOT/'output/surface_support/core.py',ROOT/'variants/surface_correspondence/model.py']
    lock={**parent,**{str(p.relative_to(ROOT)):sha(p) for p in own}}
    assert all(sha(ROOT/p)==v for p,v in lock.items());write(REPORT/'input_lock.json',lock)
    z=np.load(ROOT/'output/surface_support/image_tracks.npz');frames=z['frames'];components=z['components']
    metadata=read(TRAIN/'transforms_train.json');lookup={int(Path(f['file_path']).stem):f for f in metadata['frames']}
    matrices=[np.array(lookup[int(n)]['transform_matrix']) for n in frames];rotations=[rotation(m) for m in matrices]
    image_stats=[];blue_masks=[]
    for i,number in enumerate(frames):
        comp=components[i];count=int(comp.max()+1);areas=[int((comp==k).sum()) for k in range(count)];gaps=[];border=[]
        for k in range(count):
            mask=comp==k;y,x=np.where(mask);border.append(int(min(x.min(),399-x.max(),y.min(),224-y.max())))
            distance=ndimage.distance_transform_edt(~mask)
            for j in range(k+1,count):gaps.append(float(distance[comp==j].min()))
        image=rgba(lookup[int(number)]).astype(np.float32)/255
        blue=(image[...,3]>.95)&(image[...,2]>1.2*np.maximum(image[...,0],image[...,1]))&(image[...,2]>.25)
        blue_masks.append(blue)
        image_stats.append({'frame':int(number),'components':count,'component_pixels':areas,'minimum_mask_gap_pixels':min(gaps) if gaps else None,
            'minimum_border_clearance_pixels':min(border) if border else None})
    pairs={};pair_stats={}
    for row in read(FLOW/'manifest.json')['pairs']:
        with np.load(FLOW/row['file']) as f:forward=torch.tensor(f['forward'],dtype=torch.float32);backward=torch.tensor(f['backward'],dtype=torch.float32)
        key=(row['source'],row['target']);pairs[key]=(forward,backward);i=int(np.where(frames==row['source'])[0][0])
        valid=forward_backward_valid(forward,backward).numpy();length=torch.linalg.vector_norm(forward,dim=-1).numpy();duck=components[i]>=0;blue=blue_masks[i]
        pair_stats[key]={'source':key[0],'target':key[1],'duck_FB_valid_fraction':float(valid[duck].mean()),
          'duck_flow_median_pixels':float(np.median(length[duck&valid])) if np.any(duck&valid) else None,
          'blue_region_FB_valid_fraction':float(valid[blue].mean()) if blue.any() else None,
          'blue_region_flow_median_pixels':float(np.median(length[blue&valid])) if np.any(blue&valid) else None}
    windows=[]
    for start,end in [(i,i+7) for i in range(113)]+[(0,4)]:
        stats=image_stats[start:end+1];reasons=[]
        if any(s['components']!=4 for s in stats):reasons.append('not_four_separated_components')
        if any(min(s['component_pixels'],default=0)<50 for s in stats):reasons.append('small_component')
        if any(s['minimum_border_clearance_pixels'] is None or s['minimum_border_clearance_pixels']<3 for s in stats):reasons.append('image_border')
        if any(s['minimum_mask_gap_pixels'] is None or s['minimum_mask_gap_pixels']<2 for s in stats):reasons.append('silhouettes_too_close')
        minimum_purity=1.;minimum_support=1.;minimum_pixels=10**9;failed=None
        if not reasons:
            labels=components[start].copy();alive=np.ones(4,bool)
            for i in range(start,end+1):
                if i>start:labels,ambiguous=propagate(labels,components[i],*pairs[(int(frames[i-1]),int(frames[i]))])
                mapping,alive=qualify_fragments(labels,components[i],alive)
                if not alive.all():reasons.append('FB_fragment_correspondence_failed');failed=int(frames[i]);break
                for k,c in enumerate(mapping):
                    region=components[i]==c;known=labels[region];known=known[known>=0]
                    minimum_pixels=min(minimum_pixels,len(known));minimum_purity=min(minimum_purity,float(np.mean(known==k)));minimum_support=min(minimum_support,len(known)/int(region.sum()))
        angle=[float(np.degrees(np.arccos(np.clip((np.trace(rotations[i]@rotations[start].T)-1)/2,-1,1)))) for i in range(start,end+1)]
        shift=[float(np.linalg.norm(matrices[i][:3,3]-matrices[start][:3,3])) for i in range(start,end+1)]
        ps=[pair_stats[(int(frames[i]),int(frames[i+1]))] for i in range(start,end)]
        windows.append({'start_index':start,'end_index':end,'frames':frames[start:end+1].tolist(),'observations':end-start+1,
          'qualified':not reasons,'reasons':reasons,'first_FB_failure_frame':failed,
          'camera_max_rotation_from_start_degrees':max(angle),'camera_max_translation_from_start_scene_units':max(shift),
          'minimum_mask_gap_pixels':min(s['minimum_mask_gap_pixels'] or 0 for s in stats),'minimum_component_pixels':min(min(s['component_pixels'],default=0) for s in stats),
          'minimum_known_label_purity':minimum_purity if failed is None and not reasons else None,'minimum_propagated_component_support_fraction':minimum_support if not reasons else None,
          'minimum_supported_pixels':minimum_pixels if not reasons else None,'pair_flow_statistics':ps})
    qualified=[w for w in windows if w['qualified'] and w['observations']==8];chosen=[];occupied=set()
    for w in sorted(qualified,key=lambda r:r['camera_max_rotation_from_start_degrees']):
        indices=set(range(w['start_index'],w['end_index']+1))
        if indices&occupied:continue
        chosen.append(w);occupied|=indices
        if len(chosen)==3:break
    shorter=windows[-1]
    if shorter['qualified']:chosen.append(shorter)
    selected=[]
    for index,w in enumerate(chosen):
        name=f"clip{index+1}_{w['frames'][0]:04d}_{w['frames'][-1]:04d}";w={**w,'name':name};selected.append(w)
        sheet=Image.new('RGB',(1600,265*int(np.ceil(len(w['frames'])/4))))
        draw=ImageDraw.Draw(sheet)
        for j,number in enumerate(w['frames']):
            image=rgba(lookup[number]);rgb=(image[...,:3].astype(float)*(image[...,3:4]/255)).round().astype('uint8')
            x=(j%4)*400;y=(j//4)*265;sheet.paste(Image.fromarray(rgb),(x,y+40));draw.text((x+8,y+10),f'original training frame {number}',fill='white')
        sheet.save(OUT/(name+'.png'))
        for number in [w['frames'][0],w['frames'][len(w['frames'])//2],w['frames'][-1]]:
            with Image.open(TRAIN/(lookup[number]['file_path']+'.png')) as im:im.convert('RGBA').save(OUT/f'native{number:04d}.png')
        if index==0:
            crops=Image.new('RGB',(800,200*len(w['frames'])),'#182a3d');dr=ImageDraw.Draw(crops)
            for j,number in enumerate(w['frames']):
                i=int(np.where(frames==number)[0][0])
                with Image.open(TRAIN/(lookup[number]['file_path']+'.png')) as native_image:native=np.array(native_image.convert('RGBA'))
                scale_x=native.shape[1]/400;scale_y=native.shape[0]/225
                rgb=(native[...,:3].astype(float)*(native[...,3:4]/255)).round().astype('uint8');full=Image.fromarray(rgb)
                for k in range(4):
                    yy,xx=np.where(components[i]==k);box=(max(0,int(xx.min()*scale_x-16)),max(0,int(yy.min()*scale_y-16)),min(native.shape[1],int((xx.max()+1)*scale_x+16)),min(native.shape[0],int((yy.max()+1)*scale_y+16)))
                    crop=full.crop(box);crop.thumbnail((195,170));crops.paste(crop,(200*k,200*j+30));dr.text((200*k+4,200*j+8),f'frame{number} component{k}',fill='white')
            crops.save(OUT/'primary_duck_crops.png')
        print('selected',w['name'],w['camera_max_rotation_from_start_degrees'],w['minimum_mask_gap_pixels'],flush=True)
    assert all(sha(ROOT/p)==v for p,v in lock.items())
    write(REPORT/'results.json',{'scene':'rubber_duck','training_frames_checked':120,'eight_observation_windows_scanned':113,
        'qualified_eight_observation_windows':len(qualified),'selected':selected,'all_windows':windows,'per_frame_stats':image_stats,
        'other_installed_PIDG_scenes':[p.name for p in sorted((ROOT/'data/PIDG').iterdir()) if p.is_dir() and p.name!='rubber_duck'],
        'other_scenes_detailed_visual_review':False,'C1_model_used_to_select':False,'new_heldout_images_used':False,'all_SHA_unchanged':True,
        'training_or_GPU_model_steps':0,'manual_review_pending':True})
    html='<!doctype html><meta charset="utf-8"><title>PIDG short clip audit</title><style>body{background:#142538;color:#eee;font:16px system-ui;margin:24px}img{max-width:100%}a{color:#8ee}</style><h1>Rubber Duck: training-only short diagnostic clips</h1><p>Selection uses silhouettes, raw FB flow and camera metadata, not C1 predictions. Separated silhouettes do not prove zero3D contact or independent geometry truth.</p><a href="../../experiments/short_clip_audit/assessment.md">Assessment</a>'
    for w in selected:html+=f"<h2>{w['name']}</h2><p>Original train frames {w['frames']}; camera rotation {w['camera_max_rotation_from_start_degrees']:.3f}deg; minimum silhouette gap {w['minimum_mask_gap_pixels']:.2f}px at400x225.</p><img src='{w['name']}.png'>"
    html+="<h2>Primary clip: native-resolution per-duck crops</h2><img src='primary_duck_crops.png'>";(OUT/'index.html').write_text(html,encoding='utf-8')
    print('DATA CLIP SCAN COMPLETE',len(qualified),flush=True)


if __name__=='__main__':main()
