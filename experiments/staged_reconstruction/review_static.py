"""Native S1 review artifacts and independent numerical evidence; no optimization."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from scipy.spatial import cKDTree
from variants.staged_reconstruction.prepare_data import ROOT, REPORT, RUNS, DATA, read, write, sha
from variants.staged_reconstruction.freeze import verify
from variants.staged_reconstruction.static_gaussian import StaticGaussian
from variants.staged_reconstruction.train_static import setup, camera, render_static


def main():
    torch.set_num_threads(4);verify()
    protocol=read(REPORT/'S1_protocol.json');directory=RUNS/'S1_static'
    checkpoint=directory/'checkpoint.pt';receipt=read(directory/'checkpoint.json')
    assert receipt['iteration']==protocol['stage_updates']==12000 and sha(checkpoint)==receipt['sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['static_only'] and state['dynamic_parameters']==0
    assert all(bool(x.isfinite().all()) for x in state['model'].values())
    for values in state['optimizer']['state'].values():
        assert all(not torch.is_tensor(x) or bool(x.isfinite().all()) for x in values.values())
    for split,count in [('val',30),('test',50)]:
        rows=read(REPORT/f'S1_{split}_12000.json');assert rows['count']==len(rows['frames'])==count
        for key,value in rows['metrics'].items():assert abs(float(np.mean([row[key] for row in rows['frames']]))-value)<1e-10
    render=setup();g=StaticGaussian(RUNS/protocol['initialization_file']);g.load_state_dict(state['model']);g.requires_grad_(False)
    meta=read(DATA/'transforms_train.json');frames={int(Path(f['file_path']).stem):f for f in meta['frames']}
    out=ROOT/'output/staged_reconstruction/S1_static';out.mkdir(parents=True,exist_ok=True)
    panel=Image.new('RGB',(1200,3*252),'#171a21');draw=ImageDraw.Draw(panel)
    with torch.no_grad():
        for j,n in enumerate([2,86,188]):
            frame=frames[n];p=render_static(render,camera(frame,meta,1600),g)
            rgb=(p['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
            alpha=(p['alpha'].squeeze().clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
            Image.fromarray(rgb).save(out/f'final_t{n:04d}.png');Image.fromarray(alpha).save(out/f'alpha_t{n:04d}.png')
            with Image.open(DATA/(frame['file_path']+'.png')) as im:
                rgba=np.array(im.convert('RGBA'),dtype=float)/255
                gt=Image.fromarray((rgba[...,:3]*rgba[...,3:4]*255).round().astype(np.uint8))
            for k,(im,label) in enumerate([(gt,'Observed RGB (ducks excluded from S1 loss)'),(Image.fromarray(rgb),'Independent static reconstruction'),(Image.fromarray(alpha).convert('RGB'),'Static alpha')]):
                panel.paste(im.resize((400,225)),(400*k,252*j+27));draw.text((400*k+5,252*j+6),f'{n:04d}: {label}',fill='white')
    panel.save(out/'S1_review.png')
    opacity=g.get_opacity.detach().cpu().numpy().ravel();world=g.get_xyz.detach().cpu().numpy()
    xyz=np.array(read(REPORT/'camera_audit.json')['metadata']['xyz'])
    distance=cKDTree(world).query(xyz)[0]
    val=read(REPORT/'S1_val_12000.json')['metrics'];gates=protocol['image_admission']
    accepted_numeric=(val['candidate_static_alpha_mean']>=gates['validation_static_alpha_mean_min']
                      and val['background_alpha_mean']<=gates['validation_background_alpha_mean_max'])
    images={str(p.relative_to(ROOT)):sha(p) for p in out.glob('*.png')}
    summary={'stage':'S1','status':'numeric_verified_pending_native_review','numeric_admission':accepted_numeric,
        'iteration':12000,'point_count':len(world),'checkpoint_sha256':sha(checkpoint),
        'model_and_optimizer_finite':True,'opacity_median':float(np.median(opacity)),
        'opacity_below_001_fraction':float(np.mean(opacity<.01)),
        'conditional_triangulated_anchor_to_center_median_scene_units':float(np.median(distance)),
        'conditional_triangulated_anchor_to_center_p95_scene_units':float(np.quantile(distance,.95)),
        'native_image_sha256':images,'validation':val,
        'test':read(REPORT/'S1_test_12000.json')['metrics'],
        'limits':['Candidate masks are appearance/geometry hypotheses, not physical IDs.',
                  'Nearest center distance is not surface reconstruction GT.',
                  'Metadata-camera S1 and legacy-camera R0 are not a matched architecture comparison.',
                  'S2-S6 remain authorized and unfinished.']}
    write(REPORT/'S1_final_verification.json',summary);verify();print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
