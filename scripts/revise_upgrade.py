"""Create a new code revision before long training, preserving previous sources/runs."""
import argparse
import shutil
from pidg_benchmark import PROJECT,read_json,write_json,runtime_hashes
from snapshot_upgrade import NAMES
from upgrade_configs import stage_configs


def revise(phase,reason):
    variant=PROJECT/f"variants/phase{phase:02d}_{NAMES[phase]}"
    if (variant/'runs/long').exists(): raise RuntimeError("Never revise a source used in long training")
    stage=read_json(variant/'stage.json')
    revision=stage.get('revision',0)+1
    archive=variant/'reports'/f"revision_{revision-1}.json"
    write_json(archive,{'stage':stage,'manifest':read_json(variant/'source_manifest.json'),
                        'config':read_json(variant/'config.json'),'reason_for_revision':reason})
    directory=f"code_r{revision}"
    shutil.copytree(PROJECT/'variants/development/code',variant/directory,
                    ignore=shutil.ignore_patterns('__pycache__','tmp_build','*.pyc'))
    write_json(variant/'source_manifest.json',{'phase':phase,'revision':revision,'files':runtime_hashes(variant/directory)})
    write_json(variant/'config.json',stage_configs()[phase])
    stage.update(code_dir=directory,revision=revision,smoke='pending',status='implemented',
                 unit_tests='passed',revision_reason=reason)
    write_json(variant/'stage.json',stage)
    return variant


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stages',nargs='+',type=int,required=True)
    parser.add_argument('--reason',required=True)
    args=parser.parse_args()
    for phase in args.stages: print(revise(phase,args.reason))
