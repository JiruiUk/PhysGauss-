"""Coverage-corrected audit: retains v1 sparse stress case unchanged."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from ..prepare_data import ROOT, write, sha
from ..adaptive_static.geometry import select_views
from .generate import camera_view,trace

DATA = ROOT/'data/derived/clean_static_synthetic/v2'
REPORT = ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-synthetic-audit',action='store_true'); args = parser.parse_args()
    if not args.enable_synthetic_audit: raise SystemExit('Synthetic audit is opt-in')
    assert not DATA.exists(), 'Do not overwrite audit data'
    width,height = 256,144; focal = width/(2*np.tan(np.radians(56)/2))
    k = np.array([[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1.]])
    views = [camera_view(2*np.pi*c/32) for c in range(32)]; times = np.arange(5)/5
    pairs = [(c,ti) for c in range(32) for ti in range(5)]
    train_pairs = [(c,ti) for c,ti in pairs if c%4 != 3 and ti != c%5]
    train_views = np.asarray([views[c] for c,ti in train_pairs]); ids = [ti*40 for c,ti in train_pairs]
    sources = [select_views(i,train_views,ids) for i in range(len(ids))]
    assert len(sources) == 96 and all(len(s) == 6 for s in sources)
    REPORT.mkdir(parents=True,exist_ok=True)
    for case,glossy in [('diffuse',False),('glossy',True)]:
        train = DATA/case/'train'; private = DATA/case/'evaluation_only'
        (train/'rgb').mkdir(parents=True); (private/'gt').mkdir(parents=True); rows = []
        for c,ti in pairs:
            view,t = views[c],float(times[ti]); observed = trace(view,k,width,height,t,glossy)
            clean = trace(view,k,width,height,t,glossy,static_only=True); name = f'c{c:02d}_t{ti}'
            np.savez_compressed(private/'gt'/f'{name}.npz',observed_rgb=observed['rgb'].astype(np.float32),
                static_rgb=clean['rgb'].astype(np.float32),static_depth=clean['depth'],static_normal=clean['normal'],
                observed_static_visible=observed['static_visible'],thin_static_visible=clean['thin_static_visible'])
            if (c,ti) in train_pairs:
                Image.fromarray((observed['rgb']*255).round().astype('uint8')).save(train/'rgb'/f'{name}.png')
                rows.append({'image':f'rgb/{name}.png','world_to_camera':view.tolist(),'time':t})
        write(train/'cameras.json',{'width':width,'height':height,'K':k.tolist(),'frames':rows})
        write(private/'cameras.json',{'width':width,'height':height,'K':k.tolist(),
            'frames':[{'camera':c,'time_index':ti,'time':float(times[ti]),'world_to_camera':views[c].tolist(),
                       'seen_camera':c%4 != 3,'in_training':(c,ti) in train_pairs} for c,ti in pairs]})
    write(REPORT/'dataset.json',{'version':2,'cases':['diffuse','glossy'],'train_frames_per_case':96,
        'factorial_camera_time_pairs_per_case':160,'novel_camera_pairs_per_case':40,
        'source_eligibility':'All96 training views have six sources under unchanged original3deg..32deg and time-span requirements.',
        'revision_reason':'Calibration-only v1 audit found0/24 views had six eligible source images; preserve v1 as sparse stress case. Increase sampling before any RGB inference/quality result, with unchanged geometry thresholds.',
        'GT_contract':'Reconstructor reads only train RGB/cameras/times through input.load; clean static/depth/normal labels remain under evaluation_only.',
        'generator_sha256':sha(Path(__file__)),'base_raytracer_sha256':sha(Path(__file__).with_name('generate.py')),
        'input_loader_sha256':sha(Path(__file__).with_name('input.py')),
        'files':{p.relative_to(ROOT).as_posix():sha(p) for p in DATA.rglob('*') if p.is_file()}})
    print('SYNTHETIC_DENSE_AUDIT_GENERATED96x2; all96 source gates pass',flush=True)


if __name__ == '__main__': main()
