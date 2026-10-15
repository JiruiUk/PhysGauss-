"""Resumable official PIDG train/render/metrics/diagnostics for one frozen config."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from pidg_benchmark import PROJECT,SCENES,completed_metrics,runtime_hashes,sha256,write_json,read_json
from prepare_constitutive_pidg import prepare

RUNS=PROJECT/'runs/pidg_constitutive'


def budget(long_run,iterations):
    if iterations<100:raise ValueError('Use at least 100 integration steps')
    if long_run and iterations!=50000:raise ValueError('Official long benchmark requires exactly 50000 iterations')
    return {'iterations':iterations,'resolution_width':1600 if long_run else 400,'seed':0,'smoke_only':not long_run}


def verified_checkpoint(run):
    sys.path.insert(0,str(PROJECT/'reproduction/scripts'))
    from training_watchdog import verified_checkpoint as verify
    return verify(run/'checkpoints')


def check_resume_protocol(metadata,expected):
    """Only a recorded exact checkpoint-I/O patch may cross a source freeze."""
    if any(metadata.get(key)!=value for key,value in expected.items() if key!='extension_hashes'):
        raise ValueError('Resume protocol/config/source/input mismatch')
    saved=metadata.get('resume_extension_hashes',metadata['extension_hashes'])
    current=expected['extension_hashes']
    if saved==current:return None
    changed={name for name in set(saved)|set(current) if saved.get(name)!=current.get(name)}
    registry=PROJECT/'experiments/checkpoint_io_hotfix.json'
    patch=read_json(registry) if registry.exists() else {}
    if changed!={'checkpoint.py'} or saved.get('checkpoint.py')!=patch.get('old_sha256') or current.get('checkpoint.py')!=patch.get('new_sha256'):
        raise ValueError('Resume protocol/config/source/input mismatch; unrecognized source change')
    return {**patch,'registry_sha256':sha256(registry)}


def execute(command,run,step,repo,env,metadata):
    for attempt in range(3):
        metadata['commands'].append(command.copy());write_json(run/'reproduction.json',metadata)
        print('RUN',step,str(run),flush=True)
        with (run/f'{step}.log').open('a',encoding='utf-8') as stream:
            stream.write('\nCOMMAND '+subprocess.list2cmdline(command)+'\n');stream.flush()
            child=subprocess.Popen(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            write_json(RUNS/'active.json',{'controller_pid':os.getpid(),'child_pid':child.pid,'step':step,'run':str(run),
                                         'experiment':metadata['experiment'],'configuration_id':metadata['configuration_id'],'started_utc':datetime.now(timezone.utc).isoformat()})
            stamp=(run/f'{step}.log').stat().st_mtime_ns;last=time.monotonic();forwarded=False;stalled=False
            while child.poll() is None:
                stop=(RUNS/'stop_queue.request').exists()
                if stop and step=='train' and not forwarded:
                    (run/'checkpoints').mkdir(exist_ok=True)
                    (run/'checkpoints/save_and_stop.request').write_text('Save full state and exit',encoding='utf-8');forwarded=True
                current=(run/f'{step}.log').stat().st_mtime_ns
                if current!=stamp:stamp=current;last=time.monotonic()
                if step=='train' and time.monotonic()-last>900 and not stop and not (run/'checkpoints/save_and_stop.request').exists():
                    checkpoint=verified_checkpoint(run)
                    if metadata['config'].get('deformation_gradient') and not checkpoint.get('extension_saved'):
                        raise RuntimeError('Stall recovery needs complete extension state; preserving live process')
                    child.kill();child.wait();stalled=True
                    metadata.setdefault('stall_restarts',[]).append({'iteration':checkpoint['iteration'],'attempt':attempt+1});break
                time.sleep(1)
        if not stalled:return child.returncode
        if attempt==2:raise RuntimeError('Training stalled after two saved-checkpoint restarts')
        if '--resume_checkpoint' not in command:command=command+['--resume_checkpoint',str(run/'checkpoints/resume.pt')]


def run(experiment,tag,iterations=300,long_run=False,resume=False):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',tag) or not re.fullmatch(r'[A-Za-z0-9_-]+',experiment):raise ValueError('Invalid config ID or run tag')
    spec=budget(long_run,iterations);config=read_json(PROJECT/f'experiments/configs/{experiment}.json');scene=config.get('scene','rubber_duck')
    if config.get('constitutive') and scene not in ('rubber_duck','balls_reaction','controlled_two_material'):
        raise ValueError('Solid constitutive loss is not allowed for this scene')
    if long_run and config['experiment']=='A0':
        existing=PROJECT/'runs/pidg_official'/scene;info,_=completed_metrics(existing)
        if any(info[key]!=spec[key] for key in spec):raise ValueError('Existing baseline protocol mismatch')
        print('REUSE original 50k baseline:',existing,flush=True);return existing
    repo=prepare();directory=RUNS/tag/scene
    source=runtime_hashes(repo);original=runtime_hashes(PROJECT/'third_party/PIDG')
    extensions={p.name:sha256(p) for p in (PROJECT/'pidg_ext').glob('*.py')}
    dataset=(PROJECT/config['dataset_path']).resolve() if config.get('dataset_path') else PROJECT/'data/PIDG'/scene
    data={name:sha256(dataset/name) for name in ('transforms_train.json','transforms_test.json','points3d.ply','audit.json','download_manifest.json')}
    physical={}
    for key in ('material_ids_file','force_surface_file','known_force_file'):
        if config.get(key):
            path=Path(config[key]);path=path if path.is_absolute() else PROJECT/path
            physical[str(path.resolve())]=sha256(path)
    if config.get('multi_material') and not config.get('material_ids_file'):raise ValueError('Known material annotations required for an official multi-material scene')
    config_id=config.get('base_config',SCENES.get(scene))
    if config_id not in SCENES.values():raise ValueError('Unknown base config')
    profile='controlled_synthetic_long' if long_run and config.get('dataset_path') else 'official' if long_run else 'controlled_synthetic_smoke' if config.get('dataset_path') else 'engineering_smoke'
    expected={**spec,'profile':profile,'scene':scene,'config':config,'source_hashes':source,'original_source_hashes':original,
              'extension_hashes':extensions,'dataset_hashes':data,'physical_input_hashes':physical,
              'evaluator_sha256':sha256(repo/'metrics.py'),'shared_checkpoint_helper_sha256':sha256(PROJECT/'reproduction/scripts/training_checkpoint.py')}
    if directory.exists():
        if not resume:raise FileExistsError(f'Existing run preserved: {directory}; use explicit resume')
        metadata=read_json(directory/'reproduction.json')
        io_patch=check_resume_protocol(metadata,expected)
        if metadata['status']=='completed':completed_metrics(directory);return directory
        checkpoint=verified_checkpoint(directory)
        if config.get('deformation_gradient') and not checkpoint.get('extension_saved'):raise ValueError('Incomplete extension checkpoint')
        if io_patch is not None:
            metadata.setdefault('checkpoint_io_hotfix_history',[]).append({**io_patch,'iteration':checkpoint['iteration'],'checkpoint_sha256':checkpoint['sha256'],'utc':datetime.now(timezone.utc).isoformat()})
            metadata['resume_extension_hashes']=extensions
        if metadata.get('error'):
            metadata.setdefault('failure_history',[]).append({'error':metadata.pop('error'),'recovered_at_iteration':checkpoint['iteration']})
        metadata.setdefault('resume_history',[]).append({'iteration':checkpoint['iteration'],'utc':datetime.now(timezone.utc).isoformat()})
    else:
        directory.mkdir(parents=True,exist_ok=False)
        metadata={**expected,'experiment':config.get('experiment',experiment),'configuration_id':experiment,'commands':[],
                  'started_utc':datetime.now(timezone.utc).isoformat(),'completed_steps':[],
                  'torch':importlib.metadata.version('torch'),'python':sys.version}
        write_json(directory/'extension_config.json',config)
    env=os.environ.copy();env['PYTHONPATH']=str(PROJECT)+os.pathsep+env.get('PYTHONPATH','')
    env['PIDG_CONSTITUTIVE_CONFIG']=str(directory/'extension_config.json')
    env['PIDG_HASH_SOURCE_DIR']=str(PROJECT/'third_party/PIDG/hashencoder');env['PIDG_HASH_BUILD_DIR']=str(PROJECT/'third_party/PIDG/tmp_build')
    for name in ('hashencoder.cu','bindings.cpp','hashencoder.h'):
        assert sha256(repo/'hashencoder/src'/name)==sha256(PROJECT/'third_party/PIDG/hashencoder/src'/name)
    prefix=[sys.executable,'-u',str(PROJECT/'scripts/pidg_variant_entrypoint.py'),'--repo',str(repo),'--seed','0']
    common=['-s',str(dataset),'-m',str(directory),'--conf',str(repo/f'arguments/pidg/{config_id}.py'),'--load2gpu_on_the_fly','--resolution',str(spec['resolution_width'])]
    train=[*prefix,'train_pidg.py',*common,'--iterations',str(iterations),'--checkpoint_interval','1000' if long_run else str(iterations),'--save_image_dir',str(directory/'diagnostics')]
    if not long_run:train+=['--warm_up','0','--densify_until_iter','0','--test_iterations',str(iterations),'--save_iterations',str(iterations)]
    if resume and 'train' not in metadata['completed_steps']:train+=['--resume_checkpoint',str(directory/'checkpoints/resume.pt')]
    commands=[('train',train),('render',[*prefix,'render.py',*common,'--iteration','best','--skip_train']),('metrics',[*prefix,'metrics.py','-m',str(directory)])]
    if config.get('identifiability'):commands.append(('diagnostics',[sys.executable,'-u',str(PROJECT/'scripts/diagnose_constitutive.py'),'--run',str(directory)]))
    try:
        for step,command in commands:
            if step in metadata['completed_steps']:continue
            if (RUNS/'stop_queue.request').exists():
                metadata['status']='saved_stopped';write_json(directory/'reproduction.json',metadata);raise SystemExit(75)
            metadata['status']={'train':'training','render':'rendering','metrics':'evaluating','diagnostics':'diagnosing'}[step]
            code=execute(command,directory,step,repo if step!='diagnostics' else PROJECT,env,metadata)
            if code==75:
                metadata['status']='saved_stopped';write_json(directory/'reproduction.json',metadata);raise SystemExit(75)
            if code:raise RuntimeError(f'{step} failed ({code}); see {directory/step}.log')
            metadata['completed_steps'].append(step);write_json(directory/'reproduction.json',metadata)
        assert runtime_hashes(repo)==source and runtime_hashes(PROJECT/'third_party/PIDG')==original
        assert {p.name:sha256(p) for p in (PROJECT/'pidg_ext').glob('*.py')}==extensions
        assert all(sha256(dataset/name)==value for name,value in data.items())
        assert all(sha256(path)==value for path,value in physical.items())
        log=(directory/'train.log').read_text(encoding='utf-8',errors='replace')
        assert 'Training complete.' in log and not re.search(r'(?:Loss=|loss = )\s*(?:nan|inf)',log,re.I)
        checkpoint=verified_checkpoint(directory);assert checkpoint['iteration']==iterations
        if config.get('deformation_gradient'):assert checkpoint.get('extension_saved')
        metadata.update(status='completed',metrics=read_json(directory/'results.json'),finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(directory/'reproduction.json',metadata);completed_metrics(directory);print(json.dumps(metadata['metrics']),flush=True)
    except Exception as error:
        metadata.update(status='failed',error=str(error));write_json(directory/'reproduction.json',metadata);raise
    return directory


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',required=True);parser.add_argument('--tag',required=True)
    parser.add_argument('--iterations',type=int);parser.add_argument('--long',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run(args.experiment,args.tag,args.iterations or (50000 if args.long else 300),args.long,args.resume)
