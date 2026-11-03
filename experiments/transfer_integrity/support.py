"""Independent reuse of the historical early-bank retention audit on new states.

This cannot verify material identity after an untracked occlusion.
"""
from pathlib import Path
import sys
import os
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'variants/association_training/code'))
os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
from variants.association_training import evaluate as e
from variants.metric_tuning.runtime import read_json,write_json,sha256
REPORT=ROOT/'experiments/transfer_integrity'


def main():
    torch.set_num_threads(4)
    original=read_json(ROOT/'experiments/association_training/training_data.json')
    assert sha256(Path(original['path']))==original['sha256']
    with np.load(original['path']) as z:labels=z['labels'].copy()
    meta=read_json(e.TRAIN/'transforms_train.json');meta['frames'].sort(key=lambda f:int(Path(f['file_path']).stem))
    frames=np.asarray([int(Path(f['file_path']).stem) for f in meta['frames']])
    with np.load(ROOT/'output/surface_support/image_tracks.npz') as z:components={int(n):v.copy() for n,v in zip(z['frames'],z['components'])}
    with np.load(ROOT/'output/surface_correspondence/fragment_tracks.npz') as z:
        assert np.array_equal(z['frames'],frames);mapping=z['component_mapping'][0].copy()
    e.OUT=ROOT/'output/transfer_integrity/support';e.OUT.mkdir(parents=True,exist_ok=True)
    e.REPORT=REPORT;e.RUNS=ROOT/'runs/pidg_transfer_integrity'
    models={'C1_original_50k':ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck',
            **{n:e.RUNS/'pilot'/n/'rubber_duck' for n in ['U0_C1_control','U1_transfer_geometry']}}
    result={}
    for name,directory in models.items():
        rows=e.evaluate_model(name,directory,meta,components,frames,labels,mapping)['frames']
        result[name]={}
        for period,low,high in [('early',1,27),('middle',28,172),('late',173,200)]:
            subset=[x for x in rows if low<=x['frame']<=high]
            result[name][period]={'samples':len(subset),'early_bank_alpha_share_on_any_duck_mean':float(np.mean([x['early_bank_alpha_share_on_any_duck'] for x in subset]))}
        result[name]['physical_ID_verified_after_occlusion']=False
    write_json(REPORT/'support_summary.json',{'models':result,'binding_sha256':original['sha256'],
               'limitation':'Visible early-bank support is not physical identity; unknown image associations remain unknown',
               'evaluator_sha256':sha256(Path(__file__)),'historical_evaluator_sha256':sha256(Path(e.__file__))})


if __name__=='__main__':main()
