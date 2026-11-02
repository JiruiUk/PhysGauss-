"""Bind row-index sidecars to the exact historical full model, never object IDs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.surface_correspondence.common import *


def main():
    result=read(REPORT/'results.json');lock=read(REPORT/'input_lock.json')
    checkpoint=DIRECTORY/'checkpoints/resume.pt';expected=sha(checkpoint)
    assert expected=='b5d4cf5c9fb8fba4fcc26124555d48aa7d3a963a8be1d28488954752db486480'
    write(REPORT/'bank_bindings.json',{'full_checkpoint':str(checkpoint.relative_to(ROOT)),'full_checkpoint_sha256':expected,
        'training_metadata_sha256':sha(TRAIN/'transforms_train.json'),'canonical_particle_union_file':'output/surface_correspondence/surface_trajectories.npz',
        'canonical_particle_union_sha256':sha(OUT/'surface_trajectories.npz'),
        'banks':{b['name']:{'file':b['bank_file'],'sha256':b['bank_sha256'],'anchor':b['anchor'],'seed_frames':b['seed_frames']} for b in result['banks']},
        'row_ID_scope':'Indices in this exact frozen C1 checkpoint only; not physical IDs or stable across training, densification, reorder or another model',
        'loader_policy':'Reject checkpoint/data/bank SHA mismatch; never apply row indices by matching point count or nearest coordinate',
        'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}})


if __name__=='__main__':main()
