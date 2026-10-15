"""Reuse verified completed original baselines; never starts or repeats training."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from pidg_benchmark import PROJECT, SCENES, completed_metrics, dataset_hashes, runtime_hashes, sha256, write_json


def audit():
    source = PROJECT / 'third_party/PIDG'
    rows = []
    for scene, config in SCENES.items():
        run = PROJECT / 'runs/pidg_official' / scene
        metadata, metrics = completed_metrics(run)
        assert (metadata['iterations'],metadata['resolution_width'],metadata['seed'],
                metadata['profile'],metadata['smoke_only']) == (50000,1600,0,'official',False)
        data_audit = __import__('json').loads((PROJECT / f'data/PIDG/{scene}/audit.json').read_text())
        assert not data_audit['missing'] and not data_audit['shape_mismatches']
        train_log = run / 'train.log'
        assert '50000/50000' in train_log.read_text(encoding='utf-8',errors='replace')
        assert all((run/name).is_file() for name in ('render.log','metrics.log','results.json'))
        rows.append({'scene':scene,'config':f'arguments/pidg/{config}.py','run':str(run),
                     'protocol':{key:metadata[key] for key in ('iterations','resolution_width','seed','profile','smoke_only')},
                     'metrics':metrics,'dataset_hashes':dataset_hashes(scene),
                     'artifact_hashes':{name:sha256(run/name) for name in ('reproduction.json','results.json','train.log','render.log','metrics.log')},
                     'train_frames':data_audit['train_count'],'test_frames':data_audit['test_count']})
    config_inventory = {family:[p.stem for p in sorted((source/'arguments'/family).glob('*.py'))]
                        for family in ('pidg','dnerf','hypernerf','neu3d')}
    return {'milestone':'M0','status':'passed_reused_baselines','retrained':False,
            'checked_at':datetime.now(timezone.utc).isoformat(),
            'upstream_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
            'baselines':rows,'current_runtime_hashes':runtime_hashes(source),'config_inventory':config_inventory,
            'unavailable_datasets':['D-NeRF','HyperNeRF','Neu3D'],
            'limitations':['Current baseline has disclosed existing release/Windows compatibility fixes.',
                          'Historical runs lack runtime source hashes; current hashes do not prove historical provenance.',
                          'No material/force recovery or future-prediction GT is established.']}


if __name__=='__main__':
    result=audit()
    write_json(PROJECT/'baseline_metrics.json',result)
    write_json(PROJECT/'experiments/baseline_source_manifest.json',result['current_runtime_hashes'])
    print('M0 PASS: reused five completed 50k official baselines; no training started.')
