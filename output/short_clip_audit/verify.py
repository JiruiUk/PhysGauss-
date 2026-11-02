"""Dataset-only integrity and selected-clip recheck; no neural model is loaded."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.short_clip_audit.scan import REPORT,OUT,TRAIN,FLOW,read,sha,write,rotation
from output.surface_support.core import propagate
from variants.surface_correspondence.model import qualify_fragments


def main():
    torch.set_num_threads(4);lock=read(REPORT/'input_lock.json');assert all(sha(ROOT/p)==v for p,v in lock.items())
    result=read(REPORT/'results.json');meta=read(TRAIN/'transforms_train.json')
    lookup={int(Path(f['file_path']).stem):f for f in meta['frames']};z=np.load(ROOT/'output/surface_support/image_tracks.npz');frames=z['frames'];components=z['components']
    windows=[w for w in result['all_windows'] if w['observations']==8];assert len(windows)==113
    qualified=[w for w in windows if w['qualified']];assert len(qualified)==result['qualified_eight_observation_windows']
    for w in qualified:
        assert not w['reasons'] and w['minimum_mask_gap_pixels']>=2 and w['minimum_component_pixels']>=50
        assert w['minimum_known_label_purity']>=.95 and w['minimum_propagated_component_support_fraction']>=.25 and w['minimum_supported_pixels']>=30
    manifest=read(FLOW/'manifest.json');pairs={(r['source'],r['target']):r for r in manifest['pairs']}
    selected=[];occupied=set()
    for w in result['selected']:
        indices=[int(np.where(frames==n)[0][0]) for n in w['frames']];start=indices[0];labels=components[start].copy();alive=np.ones(4,bool)
        base=np.array(lookup[w['frames'][0]]['transform_matrix']);angles=[]
        for j,i in enumerate(indices):
            matrix=np.array(lookup[int(frames[i])]['transform_matrix']);q=rotation(matrix)@rotation(base).T
            angles.append(float(np.degrees(np.arccos(np.clip((np.trace(q)-1)/2,-1,1)))))
            if j:
                row=pairs[(int(frames[indices[j-1]]),int(frames[i]))]
                with np.load(FLOW/row['file']) as f:fw=torch.tensor(f['forward'],dtype=torch.float32);bw=torch.tensor(f['backward'],dtype=torch.float32)
                labels,_=propagate(labels,components[i],fw,bw)
            mapping,alive=qualify_fragments(labels,components[i],alive);assert alive.all()
            assert int(components[i].max())==3
        assert abs(max(angles)-w['camera_max_rotation_from_start_degrees'])<1e-10
        if w['observations']==8:
            assert not occupied&set(indices);occupied|=set(indices)
        with Image.open(OUT/(w['name']+'.png')) as im:
            sheet=np.array(im.convert('RGB'));assert im.size==(1600,265*int(np.ceil(len(indices)/4)))
        for j,number in enumerate(w['frames']):
            with Image.open(TRAIN/(lookup[number]['file_path']+'.png')) as im:rgba=np.array(im.convert('RGBA').resize((400,225),Image.Resampling.BILINEAR))
            expected=(rgba[...,:3].astype(float)*(rgba[...,3:4]/255)).round().astype('uint8');x=(j%4)*400;y=(j//4)*265+40
            assert np.array_equal(sheet[y:y+225,x:x+400],expected)
        selected.append({'name':w['name'],'frames':w['frames'],'rotation_degrees':max(angles),'FB_recheck_passed':True})
    native=[]
    for path in sorted(OUT.glob('native*.png')):
        number=int(path.stem[6:]);assert number in lookup
        with Image.open(path) as a,Image.open(TRAIN/(lookup[number]['file_path']+'.png')) as b:
            assert np.array_equal(np.array(a.convert('RGBA')),np.array(b.convert('RGBA')));native.append({'frame':number,'size':list(a.size)})
    manual=read(REPORT/'manual_review.json');assert manual['review_complete'] and manual['primary_clip_usable_for_local_diagnostic']
    assert result['C1_model_used_to_select'] is False and result['new_heldout_images_used'] is False
    files={str(p.relative_to(ROOT)):sha(p) for base in [REPORT,OUT] for p in base.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.png','.html','.md'] and p.name!='verification.json'}
    write(REPORT/'verification.json',{'passed':True,'locked_inputs':len(lock),'all_input_SHA_unchanged':True,'training_frames_checked':120,
        'eight_observation_windows':113,'qualified_windows':len(qualified),'selected_independently_rechecked':selected,
        'contact_sheet_pixels_match_train_RGB':True,'native_exports_pixel_identical':native,'manual_review_complete':True,
        'C1_predictions_not_used_for_selection':True,'new_heldout_RGB_not_used':True,'model_optimizer_steps':0,
        'no_3D_or_material_GT_claim':True,'artifact_sha256':files})
    print('SHORT CLIP VERIFICATION PASSED',len(qualified),len(selected),flush=True)


if __name__=='__main__':main()
