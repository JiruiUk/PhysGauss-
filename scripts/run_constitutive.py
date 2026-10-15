"""One independent official PIDG short integration run with extension config."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from pidg_benchmark import PROJECT,SCENES,completed_metrics,dataset_hashes,runtime_hashes,sha256,write_json
from prepare_constitutive_pidg import prepare


def run(experiment,tag,iterations=300):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',tag):raise ValueError('Invalid run tag')
    config=json.loads((PROJECT/f'experiments/configs/{experiment}.json').read_text())
    scene=config.get('scene','rubber_duck')
    if config.get('constitutive') and scene not in ('rubber_duck','balls_reaction','controlled_two_material'):
        raise ValueError('Solid constitutive loss is not allowed for this scene')
    repo=prepare();run=PROJECT/'runs/pidg_constitutive'/tag/scene
    run.mkdir(parents=True,exist_ok=False)
    write_json(run/'extension_config.json',config)
    source=runtime_hashes(repo);original=runtime_hashes(PROJECT/'third_party/PIDG')
    extension_hashes={p.name:sha256(p) for p in (PROJECT/'pidg_ext').glob('*.py')}
    dataset=(PROJECT/config['dataset_path']).resolve() if config.get('dataset_path') else PROJECT/'data/PIDG'/scene
    if config.get('dataset_path'):
        data={name:sha256(dataset/name) for name in ('transforms_train.json','transforms_test.json','points3d.ply','audit.json','download_manifest.json')}
    else:data=dataset_hashes(scene)
    if config.get('multi_material'):
        path=Path(config['material_ids_file']);path=path if path.is_absolute() else PROJECT/path
        data['material_ids']=sha256(path)
    env=os.environ.copy();env['PYTHONPATH']=str(PROJECT)+os.pathsep+env.get('PYTHONPATH','')
    env['PIDG_CONSTITUTIVE_CONFIG']=str(run/'extension_config.json')
    env['PIDG_HASH_SOURCE_DIR']=str(PROJECT/'third_party/PIDG/hashencoder')
    env['PIDG_HASH_BUILD_DIR']=str(PROJECT/'third_party/PIDG/tmp_build')
    for name in ('hashencoder.cu','bindings.cpp','hashencoder.h'):
        assert sha256(repo/'hashencoder/src'/name)==sha256(PROJECT/'third_party/PIDG/hashencoder/src'/name)
    entry=PROJECT/'scripts/pidg_variant_entrypoint.py'
    prefix=[sys.executable,'-u',str(entry),'--repo',str(repo),'--seed','0']
    config_id=config.get('base_config',SCENES.get(scene))
    if config_id not in SCENES.values():raise ValueError('Unknown base config')
    common=['-s',str(dataset),'-m',str(run),'--conf',str(repo/f'arguments/pidg/{config_id}.py'),'--load2gpu_on_the_fly','--resolution','400']
    metadata={'experiment':experiment,'scene':scene,'iterations':iterations,'resolution_width':400,'seed':0,
              'profile':'controlled_synthetic_smoke' if config.get('dataset_path') else 'engineering_smoke','smoke_only':True,'started_utc':datetime.now(timezone.utc).isoformat(),
              'source_hashes':source,'original_source_hashes':original,'extension_hashes':extension_hashes,
              'dataset_hashes':data,'config':config,'commands':[]}
    stages=[('train',[*prefix,'train_pidg.py',*common,'--iterations',str(iterations),'--warm_up','0','--densify_until_iter','0','--test_iterations',str(iterations),'--save_iterations',str(iterations),'--checkpoint_interval',str(iterations),'--save_image_dir',str(run/'diagnostics')]),
            ('render',[*prefix,'render.py',*common,'--iteration','best','--skip_train']),
            ('metrics',[*prefix,'metrics.py','-m',str(run)])]
    try:
        for step,command in stages:
            metadata['status']={'train':'training','render':'rendering','metrics':'evaluating'}[step]
            metadata['commands'].append(command);write_json(run/'reproduction.json',metadata)
            print('RUN',step,str(run),flush=True)
            with (run/f'{step}.log').open('w',encoding='utf-8') as stream:
                code=subprocess.call(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if code==75:
                metadata['status']='saved_stopped';write_json(run/'reproduction.json',metadata);raise SystemExit(75)
            if code:raise RuntimeError(f'{step} failed ({code}); see {run/step}.log')
        assert runtime_hashes(repo)==source and runtime_hashes(PROJECT/'third_party/PIDG')==original
        assert {p.name:sha256(p) for p in (PROJECT/'pidg_ext').glob('*.py')}==extension_hashes
        assert all(sha256(dataset/name)==value for name,value in data.items() if name!='material_ids')
        if config.get('multi_material'):assert sha256(path)==data['material_ids']
        log=(run/'train.log').read_text(encoding='utf-8',errors='replace')
        assert 'Training complete.' in log and not re.search(r'(?:Loss=|loss = )\s*(?:nan|inf)',log,re.I)
        metadata.update(status='completed',metrics=json.loads((run/'results.json').read_text()),finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(run/'reproduction.json',metadata);completed_metrics(run)
        sys.path.insert(0,str(PROJECT/'reproduction/scripts'))
        from training_watchdog import verified_checkpoint
        assert verified_checkpoint(run/'checkpoints')['iteration']==iterations
        print(json.dumps(metadata['metrics']),flush=True)
    except Exception as error:
        metadata.update(status='failed',error=str(error));write_json(run/'reproduction.json',metadata);raise
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',required=True)
    parser.add_argument('--tag',required=True);parser.add_argument('--iterations',type=int,default=300)
    args=parser.parse_args();run(args.experiment,args.tag,args.iterations)
