"""One bounded C1 local-shape candidate, with explicit preflight and graph-review gates."""
import argparse
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.color_repair import run as parent
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from .prepare import prepare
HERE=Path(__file__).parent
RUNS=ROOT/'runs/pidg_shape_coherence';REPORTS=ROOT/'experiments/shape_coherence'
OUT=ROOT/'output/shape_coherence';NAME='P1_local_shape'
CONTROL=parent.RUNS/'C1_repaired/rubber_duck'


def configuration(smoke=False,off=False):
    c,text=parent.configuration('C1_repaired',False,off)
    c.update(shape_enabled=not off,shape_affine_weight=1e-5,shape_collapse_weight=1e-6,
             shape_start=50000 if smoke else 16000,shape_ramp=10 if smoke else 2000,
             shape_interval=1 if smoke else 20,shape_samples=128,shape_min_stretch=.3,shape_max_stretch=3.,
             shape_audit=smoke)
    assert c['affine_weight']==0 and not c['coverage_density'] and c['kinematic_weight']==0
    return c,text


def freeze(repo):
    files=[*HERE.glob('*.py'),ROOT/'tests/test_shape_coherence.py',REPORTS/'plan.json']
    for folder in ['training','final_test']:
        files.extend(p for p in (base.DATA/folder).rglob('*') if p.is_file())
    return {'parent':parent.freeze(parent.HERE/'code'),'code':engine.runtime_hashes(repo),
            'files':{str(p.relative_to(ROOT)):engine.sha256(p) for p in sorted(set(files))},
            'full_3k_sha256':engine.sha256(parent.RUNS/'shared_prefix/rubber_duck/checkpoints/resume.pt')}


def assert_frozen(repo):
    assert engine.read_json(REPORTS/'source_freeze.json')==freeze(repo),'Shape source/data changed'


def wire(repo):
    engine.RUNS,engine.REPORTS,engine.repo_global,engine.assert_frozen=RUNS,REPORTS,repo,assert_frozen
    base.RUNS,base.REPORTS,base.assert_frozen=RUNS,REPORTS,assert_frozen


def initialize(name,smoke=False,off=False):
    path=RUNS/name/'rubber_duck';c,text=configuration(smoke,off)
    if (path/'metric_config.json').exists():
        assert engine.read_json(path/'metric_config.json')==c and (path/'model_config.py').read_text()==text
        return path
    path.mkdir(parents=True,exist_ok=True)
    engine.write_json(path/'metric_config.json',c);engine.write_text_atomic(path/'model_config.py',text)
    engine.write_json(path/'reproduction.json',{'id':name,'commands':[],'status':'initialized',
                      'target_iterations':20 if off else 50100 if smoke else 50000,'historical_control_reused':True})
    return path


def branch(destination,smoke=False):
    if (destination/'branch.json').exists():return
    source=CONTROL if smoke else parent.RUNS/'shared_prefix/rubber_duck'
    boundary=50000 if smoke else 3000
    receipt=engine.read_json(source/'checkpoints/resume.json')
    assert receipt['iteration']==boundary and receipt['sha256']==engine.sha256(source/'checkpoints/resume.pt')
    for folder in ['point_cloud','deform','checkpoints']:shutil.copytree(source/folder,destination/folder,dirs_exist_ok=True)
    for file in ['metric_best.json','metric_history.jsonl']:shutil.copyfile(source/file,destination/file)
    c=engine.read_json(destination/'metric_config.json');checkpoint=destination/'checkpoints/resume.pt'
    original_sha=engine.sha256(checkpoint);assert original_sha==receipt['sha256']
    if smoke:
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['opt']['iterations']==50000
        state['opt']['iterations']=50100
        torch.save(state,checkpoint)
        receipt={**receipt,'path':str(checkpoint),'sha256':engine.sha256(checkpoint),'size_bytes':checkpoint.stat().st_size}
        engine.write_json(checkpoint.with_suffix('.json'),receipt)
        # No model, optimizer or RNG tensors changed; only the preflight horizon.
    sidecar=checkpoint.with_suffix('.metrics.json');s=engine.read_json(sidecar)
    s.update(config=c,checkpoint_sha256=receipt['sha256'])
    if smoke:
        side=checkpoint.with_suffix('.shape.pt')
        torch.save({'iteration':boundary,'checkpoint_sha256':receipt['sha256'],'graph':None},side)
        s['shape_graph_sha256']=engine.sha256(side)
    engine.write_json(sidecar,s)
    engine.write_json(destination/'branch.json',{'source':str(source),'iteration':boundary,
        'destination':str(destination),'identical_full_checkpoint_sha256':receipt['sha256'],
        'original_source_checkpoint_sha256':original_sha,'branch_checkpoint_sha256':receipt['sha256'],
        'tensor_optimizer_RNG_bytes_unchanged':not smoke,'preflight_opt_iterations_only_expansion':smoke,
        'additional_training_budget':100 if smoke else 47000})


def train_smoke(path,repo,deadline,limit):
    command=[*engine.command_prefix(repo),'train_pidg.py','-s',str(base.DATA/'training'),'-m',str(path),
        '--conf',str(path/'model_config.py'),'--load2gpu_on_the_fly','--resolution','400',
        '--save_image_dir',str(path/'diagnostics'),
        '--iterations','50100','--checkpoint_interval','10','--save_iterations','50100',
        '--test_iterations','50100','--resume_checkpoint',str(path/'checkpoints/resume.pt')]
    env=engine.env_for(path)
    if limit:env['PIDG_TUNING_STOP_AFTER']=str(limit)
    return engine.execute(command,path,'smoke',env,deadline)


def preflight(repo,deadline):
    update=engine.read_json(REPORTS/'optimizer_update.json')
    assert update['status']=='passed' and all(v>0 for v in update['actual_parameter_L1_changes'].values())
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_shape_coherence.py'],cwd=ROOT,capture_output=True,text=True)
    engine.write_text_atomic(REPORTS/'tests.log',result.stdout+result.stderr)
    assert result.returncode==0
    off=initialize('_flags_off',off=True)
    if parent.step(off)<20:assert base.train(off,repo,deadline,off=True)==0
    candidate=initialize('_smoke_P1',smoke=True);branch(candidate,smoke=True)
    if parent.step(candidate)<50070:assert train_smoke(candidate,repo,deadline,50070)==75
    if parent.step(candidate)<50100:assert train_smoke(candidate,repo,deadline,0)==0
    graph=engine.read_json(candidate/'checkpoints/resume.metrics.json')['shape_graph_sha256']
    assert graph==engine.sha256(candidate/'checkpoints/resume.shape.pt')
    events=[json.loads(line) for line in (candidate/'shape_loss.jsonl').read_text().splitlines()]
    losses=[e for e in events if 'weighted_loss' in e]
    assert len(losses)==100 and all(torch.isfinite(torch.tensor(e['weighted_loss'])) for e in losses)
    assert sum(e.get('event')=='graph_built' for e in events)==1
    engine.write_json(REPORTS/'preflight.json',{'status':'passed','unit_tests_log':'tests.log','flags_off_20':True,
        'GPU_smoke_100':True,'full_resume_50070_to_50100':True,'shape_evaluations':100,
        'graph_built_once_across_resume':True,'new_parameters':0,'new_optimizer_groups':0,
        'actual_optimizer_update_sha256':engine.sha256(REPORTS/'optimizer_update.json'),
        'source_freeze_sha256':engine.sha256(REPORTS/'source_freeze.json')})


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--hours',type=float,default=12)
    args=p.parse_args();repo=prepare();wire(repo)
    RUNS.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    if not (REPORTS/'source_freeze.json').exists():engine.write_json(REPORTS/'source_freeze.json',freeze(repo))
    assert_frozen(repo);deadline=datetime.now(timezone.utc)+timedelta(hours=args.hours)
    state={'status':'preflight','pid':os.getpid(),'started_utc':engine.now()}
    with engine.lock():
        engine.write_json(RUNS/'controller.json',state)
        try:
            if not (REPORTS/'preflight.json').exists():preflight(repo,deadline)
            receipt=engine.read_json(REPORTS/'preflight.json')
            assert receipt['status']=='passed' and receipt['source_freeze_sha256']==engine.sha256(REPORTS/'source_freeze.json')
            if args.preflight_only:state['status']='preflight_complete';return
            candidate=initialize(NAME);branch(candidate)
            state['status']='warmup_to_graph_review';engine.write_json(RUNS/'controller.json',state)
            if parent.step(candidate)<16000:assert base.train(candidate,repo,deadline,limit=16000)==75
            gate=REPORTS/'candidate_graph_review.json'
            if not gate.exists():
                state['status']='awaiting_manual_graph_review';return
            review=engine.read_json(gate)
            assert review['approved'] and review['graph_sha256']==engine.sha256(candidate/'shape_reference/reference_graph.pt')
            state['status']='training';engine.write_json(RUNS/'controller.json',state)
            if parent.step(candidate)<50000:assert base.train(candidate,repo,deadline)==0
            state['status']='trained_pending_result_verification'
        except BaseException as error:
            state.update(status='failed',error=str(error));raise
        finally:
            state['updated_utc']=engine.now();engine.write_json(RUNS/'controller.json',state)


if __name__=='__main__':main()
