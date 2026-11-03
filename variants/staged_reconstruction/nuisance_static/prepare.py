"""RGB-derived synthetic initialization and support; never loads evaluator GT."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from ..prepare_data import ROOT,read,write,sha
from ..clean_static.prepare import verify as verify_parent
from ..synthetic_audit.input import load
from output.short_clip_diagnostic.camera_math import project

REPORT = ROOT/'experiments/staged_reconstruction/S1_nuisance/v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_nuisance/v1'
OUT = ROOT/'output/staged_reconstruction/S1_nuisance/v1'


def verify():
    verify_parent()
    for p,h in read(REPORT/'source_freeze.json').items(): assert sha(ROOT/p) == h,p


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-nuisance',action='store_true'); args = parser.parse_args()
    if not args.enable_nuisance: raise SystemExit('Nuisance reconstruction is opt-in')
    assert not (REPORT/'preparation.json').exists(), 'Preserve prepared inputs'
    assert not (RUNS/'stop.request').exists(), 'Saved user stop; do not resume'
    verify_parent(); protocol = read(REPORT.parent/'protocol.json')
    RUNS.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    records,inputs = {},{}
    for case in protocol['synthetic_cases']:
        source = ROOT/f'runs/pidg_staged_reconstruction/S1_clean_visibility/synthetic_v2/{case}/reconstruction.npz'
        receipt = read(ROOT/f'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2/{case}_inference.json')
        assert sha(source) == receipt['reconstruction_sha256']
        directory = ROOT/protocol['synthetic_data']/case/'train'; values = load(directory)
        for path in directory.rglob('*'):
            if path.is_file(): inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        inputs[source.relative_to(ROOT).as_posix()] = sha(source)
        with np.load(source) as data: xyz,rgb,support = data['xyz'].copy(),data['rgb'].copy(),data['support'].copy()
        # Both arms use the same unfiltered RGB reconstruction. No GT selection.
        order = np.lexsort((np.arange(len(xyz)),-support.sum(1)))
        _,first = np.unique(np.floor(xyz[order]/.06).astype(int),axis=0,return_index=True)
        ids = np.sort(order[first])
        if len(ids) > 4000: ids = np.sort(np.random.default_rng(20261103).choice(ids,4000,replace=False))
        output = RUNS/case; output.mkdir(exist_ok=True)
        np.savez_compressed(output/'initialization.npz',xyz=xyz[ids],rgb=rgb[ids],spacing=np.asarray(.04),point_ids=ids)
        masks = []
        for i,view in enumerate(values['views']):
            height,width = values['rgb'].shape[1:3]; mask = np.zeros((height,width),np.uint8)
            uv,z = project(xyz,view,values['K'])
            valid = support[:,i]&(z > .2)&np.isfinite(uv).all(1)&(uv[:,0] >= 1)&(uv[:,0] < width-1)&(uv[:,1] >= 1)&(uv[:,1] < height-1)
            for x,y in np.rint(uv[valid]).astype(int): cv2.circle(mask,(int(x),int(y)),1,1,-1)
            masks.append(mask.astype(bool))
        np.savez_compressed(output/'support.npz',reliable=np.asarray(masks))
        records[case] = {'points':len(ids),'views':len(masks),'inferred_support_pixels':int(np.asarray(masks).sum()),
            'initialization_sha256':sha(output/'initialization.npz'),'support_sha256':sha(output/'support.npz')}
    write(REPORT/'preparation.json',{'RGB_only':True,'evaluator_GT_read':False,'records':records,'inputs':inputs,
        'details':{'voxel_spacing':.06,'max_points':4000,'model_spacing':.04,'seed':20261103,
                   'support_is_train_inferred_conditional_not_GT':True,'quarantine_not_bundled_into_this_candidate':True}})
    print('NUISANCE_PREPARED',records,flush=True)


if __name__ == '__main__': main()
