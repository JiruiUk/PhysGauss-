"""Audit this run's own early bank; no old C1 rows or post-gap physical-ID claim."""
from pathlib import Path
import sys
import os
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'variants/association_training/code'))
os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder');os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
from variants.association_training import evaluate as e
from variants.scratch_integrity.prefix import REPORT,RUNS
from variants.scratch_integrity.run import NAMES
from variants.metric_tuning.runtime import read_json,write_json,sha256


def main():
    torch.set_num_threads(4);b=read_json(REPORT/'data_binding.json');assert sha256(Path(b['path']))==b['sha256']
    with np.load(b['path']) as z:labels=z['early_labels'].copy()
    meta=read_json(e.TRAIN/'transforms_train.json');meta['frames'].sort(key=lambda f:int(Path(f['file_path']).stem))
    frames=np.array([int(Path(f['file_path']).stem) for f in meta['frames']])
    with np.load(ROOT/'output/surface_support/image_tracks.npz') as z:components={int(n):c.copy() for n,c in zip(z['frames'],z['components'])}
    with np.load(ROOT/'output/surface_correspondence/fragment_tracks.npz') as z:
        assert np.array_equal(z['frames'],frames);mapping=z['component_mapping'][0].copy()
    e.OUT=ROOT/'output/scratch_integrity/support';e.OUT.mkdir(parents=True,exist_ok=True);e.REPORT=REPORT;e.RUNS=RUNS
    models={'common_prefix':Path(b['source_directory']),**{n:RUNS/'pilot'/n/'rubber_duck' for n in NAMES}};result={}
    for name,directory in models.items():
        rows=e.evaluate_model(name,directory,meta,components,frames,labels,mapping)['frames'];result[name]={}
        for period,lo,hi in [('early',1,27),('middle',28,172),('late',173,200)]:
            group=[r for r in rows if lo<=r['frame']<=hi]
            result[name][period]={'samples':len(group),'early_bank_alpha_share_on_any_duck_mean':float(np.mean([r['early_bank_alpha_share_on_any_duck'] for r in group]))}
        result[name]['physical_ID_verified_after_occlusion']=False
    write_json(REPORT/'support_summary.json',{'models':result,'own_fresh_binding_sha256':b['sha256'],'own_early_bank_count':int((labels>=0).sum()),
               'own_group_counts':[int((labels==i).sum()) for i in range(4)],'adapter_sha256':sha256(Path(__file__)),
               'limitation':'Partial own early-bank visible support, not physical identity; unknown cross-gap associations remain unknown'})


if __name__=='__main__':main()
