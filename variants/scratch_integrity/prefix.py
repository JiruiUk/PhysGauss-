"""Fresh C1-only common prefix; never reads a historical trained checkpoint."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse
import sys
import os
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.metric_tuning import run as engine
from variants.color_repair.prepare import prepare
from variants.v3_iso_scratch.run import validate_initial_state

REPORT=ROOT/'experiments/scratch_integrity'
RUNS=ROOT/'runs/pidg_scratch_integrity'
DATA=ROOT/'data/derived/kinematic_resplit/training'
PREFIX=RUNS/'common_prefix/rubber_duck'
RECIPE=ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck'


def plan():return engine.read_json(REPORT/'plan.json')


def wire(repo):
    engine.RUNS,engine.REPORTS,engine.repo_global=RUNS,REPORT,repo


def initialize():
    PREFIX.mkdir(parents=True,exist_ok=True)
    c=engine.read_json(RECIPE/'metric_config.json')
    assert not c['coverage_density'] and c['affine_weight']==0 and not c['parent_run']
    text=(RECIPE/'model_config.py').read_text(encoding='utf-8')
    import runpy
    assert runpy.run_path(str(RECIPE/'model_config.py'))['densify_until_iter']==15000
    if (PREFIX/'metric_config.json').exists():
        assert engine.read_json(PREFIX/'metric_config.json')==c
        assert (PREFIX/'model_config.py').read_text(encoding='utf-8')==text
    else:
        engine.write_json(PREFIX/'metric_config.json',c)
        (PREFIX/'model_config.py').write_text(text,encoding='utf-8')
        engine.write_json(PREFIX/'reproduction.json',{'id':'common_prefix','commands':[],'status':'initialized',
                          'seed':plan()['seed'],'warm_start':False,'pretrained_model_used':False,'target_iteration':50000})


def step(directory=PREFIX):
    p=directory/'checkpoints/resume.json'
    return engine.read_json(p)['iteration'] if p.exists() else -1


def train(directory,repo,stop,deadline,capture=False):
    prefix=engine.command_prefix(repo);prefix[-1]=str(plan()['seed'])
    command=[*prefix,'train_pidg.py','-s',str(DATA),'-m',str(directory),'--conf',str(directory/'model_config.py'),
             '--load2gpu_on_the_fly','--resolution','1600','--iterations','50001','--checkpoint_interval','1000',
             '--save_image_dir',str(directory/'diagnostics'),'--save_iterations','15000','50000',
             '--test_iterations',*map(str,range(5000,50001,5000))]
    checkpoint=directory/'checkpoints/resume.pt'
    if checkpoint.exists():
        assert not capture
        assert engine.sha256(checkpoint)==engine.read_json(checkpoint.with_suffix('.json'))['sha256']
        command+=['--resume_checkpoint',str(checkpoint)]
    env=engine.env_for(directory);env['PIDG_TUNING_STOP_AFTER']=str(stop)
    if capture:env['PIDG_CAPTURE_INITIAL_ONLY']='1'
    else:env.pop('PIDG_CAPTURE_INITIAL_ONLY',None)
    return engine.execute(command,directory,'long',env,deadline)


def freeze(repo):
    paths=[Path(__file__),REPORT/'plan.json',RECIPE/'metric_config.json',RECIPE/'model_config.py',
           ROOT/'reproduction/scripts/training_checkpoint.py',ROOT/'scripts/pidg_variant_entrypoint.py',
           DATA/'points3d.ply',DATA/'transforms_train.json',DATA/'transforms_test.json']
    # Explicitly no historical resume/PLY/deformation state is an input.
    value={'baseline_runtime':engine.runtime_hashes(repo),'inputs':{str(p.relative_to(ROOT)):engine.sha256(p) for p in paths}}
    p=REPORT/'prefix_freeze.json'
    if p.exists():assert engine.read_json(p)==value,'Common prefix inputs changed'
    else:engine.write_json(p,value)


def run(boundary=15000,hours=12):
    torch.set_num_threads(4);repo=prepare();wire(repo);initialize();freeze(repo)
    deadline=datetime.now(timezone.utc)+timedelta(hours=hours)
    engine.write_json(RUNS/'controller.json',{'status':'common_prefix','pid':os.getpid(),'target':boundary,'seed':plan()['seed'],'started_utc':engine.now()})
    if step()<0:
        assert train(PREFIX,repo,0,deadline,capture=True)==75
        zero=PREFIX/'checkpoints/resume.pt'
        state=torch.load(zero,map_location='cpu',weights_only=False);validate_initial_state(state)
        import shutil
        keep=RUNS/'random_initial_state';keep.mkdir(parents=True,exist_ok=True)
        for p in zero.parent.glob('resume.*'):shutil.copyfile(p,keep/p.name)
        engine.write_json(REPORT/'initialization.json',{'iteration':0,'optimizer_updates':0,'pretrained_model_used':False,
                          'seed':plan()['seed'],'point_count':len(state['gaussian_parameters']['_xyz']),
                          'random_initial_checkpoint_sha256':engine.sha256(zero),'fresh_capture_command':engine.read_json(PREFIX/'reproduction.json')['commands'][0]})
    if step()<boundary:assert train(PREFIX,repo,boundary,deadline)==75
    assert step()==boundary
    freeze(repo)
    engine.write_json(RUNS/'controller.json',{'status':'common_prefix_ready','iteration':boundary,'pid':os.getpid(),'updated_utc':engine.now()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--boundary',type=int,default=15000);a=p.parse_args();run(a.boundary)
