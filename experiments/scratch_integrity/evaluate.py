"""Read-only final cold-start comparison with the established native evaluator."""
from pathlib import Path
import importlib.util
import sys
import os
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('v1_measurement',ROOT/'experiments/transfer_integrity/evaluate.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
from variants.scratch_integrity.run import NAMES,configuration,freeze,prepare_code
from variants.scratch_integrity.prefix import REPORT,RUNS,DATA,PREFIX,plan
from variants.scratch_integrity.runtime import Tuning
from variants.metric_tuning.runtime import read_json,write_json,sha256
OUT=ROOT/'output/scratch_integrity'
MODELS={'C1_original_50k':ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck',
        **{n:RUNS/'pilot'/n/'rubber_duck' for n in NAMES}}
e.r.REPORT=REPORT;e.r.OUT=OUT;e.r.MODELS=MODELS


def proof():
    freeze(prepare_code());b=read_json(REPORT/'data_binding.json');start=b['source_iteration']
    assert sha256(Path(b['path']))==b['sha256']
    source=Path(b['source_directory'])
    assert sha256(source/'checkpoints/resume.pt')==b['source_checkpoint_sha256']==sha256(PREFIX/'checkpoints/resume.pt')
    assert read_json(REPORT/'initialization.json')['pretrained_model_used'] is False
    configs=[read_json(MODELS[n]/'metric_config.json') for n in NAMES]
    assert {k for k in configs[0] if configs[0][k]!=configs[1][k]}=={'transfer_integrity_enabled'}
    samples=[[json.loads(x) for x in (MODELS[n]/'camera_sampling.jsonl').read_text().splitlines()] for n in NAMES]
    assert samples[0]==samples[1] and [x['iteration'] for x in samples[0]]==list(range(start+1,50001))
    assert len((MODELS[NAMES[1]]/'transfer_loss.jsonl').read_text().splitlines())==(50000-start)//plan()['interval']
    assert not (MODELS[NAMES[0]]/'transfer_loss.jsonl').exists()
    common=torch.load(source/'checkpoints/resume.pt',map_location='cpu',weights_only=False);states={}
    for name,directory in MODELS.items():
        g,d,p=e.select(directory);assert p['iteration']==50000
        if name in NAMES:
            s=torch.load(directory/'checkpoints/resume.pt',map_location='cpu',weights_only=False)
            assert torch.equal(s['gaussian_buffers']['particle_indices'],common['gaussian_buffers']['particle_indices'])
            assert len(s['gaussian_parameters']['_xyz'])==int(b['point_count'])
        states[name]=p;del g,d;torch.cuda.empty_cache()
    write_json(REPORT/'training_verification.json',{'states':states,'activation':start,'same_post_prefix_camera_samples':len(samples[0]),
               'only_main_flag_differs':True,'actual_total_updates_each':50000,'fresh_zero_optimizer_state_proven':True,
               'row_topology_preserved_after_binding':True,'historical_trained_states_used_for_cold_arms':False})


@torch.no_grad()
def constraints():
    b=read_json(REPORT/'data_binding.json')
    models={'common_prefix':Path(b['source_directory']),**{n:MODELS[n] for n in NAMES}}
    for name,directory in models.items():
        g,d,p=e.select(directory);folder=RUNS/'evaluation'/name;folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'metric_config.json',configuration(True,b))
        t=Tuning(folder/'metric_config.json',folder);t.transfer_deform=d;t.initialize_data(DATA);rows=[]
        for i in range(1,121):
            _,terms=t.geometry_loss(b['source_iteration']+i*10,g)
            rows.append({k:float(v) if torch.is_tensor(v) else v for k,v in terms.items()})
        write_json(REPORT/f'{name}_constraints.json',{'full_state':p,'samples':120,'frames':rows,'training_constraint_diagnostic_only':True,
                   'means':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ['static_sample_time_frame','track_frame']}})
        del g,d,t;torch.cuda.empty_cache()


def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);proof()
    protected={str(p/'checkpoints/resume.pt'):sha256(p/'checkpoints/resume.pt') for p in MODELS.values()}
    e.observations()
    for n in MODELS:
        for split in ['val','test']:
            p=REPORT/f'{n}_{split}.json';v=read_json(p);v['adapter_sha256']=sha256(Path(__file__));write_json(p,v)
    constraints();sys.argv=[sys.argv[0]];e.r.main()
    assert all(sha256(Path(p))==h for p,h in protected.items());freeze(prepare_code())
    write_json(REPORT/'evaluation_adapter.json',{'wrapper_sha256':sha256(Path(__file__)),
               'measurement_helper_sha256':sha256(ROOT/'experiments/transfer_integrity/evaluate.py'),'checkpoints_unchanged':protected,
               'native_renderer_helper_sha256':sha256(ROOT/'output/fixed_camera_temporal/render_sequences.py')})


if __name__=='__main__':main()
