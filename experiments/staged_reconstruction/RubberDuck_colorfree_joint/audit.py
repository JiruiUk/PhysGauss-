"""Independent CPU replay of source composition, complete Adam and sampling."""
import random
from pathlib import Path
import torch
from variants.staged_reconstruction.prepare_data import ROOT,read,sha
from variants.staged_reconstruction.joint_rgb.core import assert_equivalent as equal

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_colorfree_joint'
RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_colorfree_joint/v1'


def audit():
    p=read(BASE/'protocol.json')
    load=lambda path:torch.load(path,map_location='cpu',weights_only=False)
    j6=load(ROOT/p['parent']);new=load(ROOT/p['static_parent']);rows={};traces=[]
    for arm in p['arms']:
        directory=RUNS/'formal'/arm;i=load(directory/'initial.pt');f=load(directory/'checkpoint.pt')
        assert sha(directory/'checkpoint.pt')==read(directory/'checkpoint.json')['sha256']
        assert i['iteration']==0 and f['iteration']==p['updates'] and len(f['sample_trace'])==p['updates']
        assert i['fusion_arm']==f['fusion_arm']==arm
        source=new['model'] if arm=='colorfree_static' else {n[7:]:v for n,v in j6['model'].items() if n.startswith('static.')}
        equal({n[7:]:v for n,v in i['model'].items() if n.startswith('static.')},source,0,0)
        for name,value in i['model'].items():
            if not name.startswith('static.'):assert torch.equal(value,j6['model'][name]),name
            if name.startswith('physics.') or (name.startswith('static.') and name[7:] not in p['static_trainable']):
                assert torch.equal(value,f['model'][name]),name
        equal(i['optimizers'][1],j6['optimizers'][1],0,0)
        equal(i['optimizers'][0],new['optimizer'] if arm=='colorfree_static' else j6['optimizers'][0],0,0)
        equal(i['physics_graphs'],j6['physics_graphs'],0,0);equal(f['physics_graphs'],i['physics_graphs'],0,0)
        assert i['bindings']==f['bindings']
        for key in ['rng_numpy','rng_torch','rng_cuda']:
            equal(i[key],j6[key],0,0);equal(f[key],i[key],0,0)
        sampler=random.Random();sampler.setstate(j6['rng_python']);stack=list(j6['camera_stack']);trace=[]
        for _ in range(p['updates']):
            if not stack:stack=sampler.sample(range(120),120)
            trace.append(stack.pop())
        assert f['sample_trace']==trace and f['camera_stack']==stack and f['rng_python']==sampler.getstate()
        counters={}
        for number,(before,after) in enumerate(zip(i['optimizers'],f['optimizers'])):
            assert before['param_groups']==after['param_groups']
            for group in after['param_groups']:
                mutable=(group['name'] in p['static_trainable']) if number==0 else group['name']!='PIDG_velocity_stress'
                for key in group['params']:
                    a=before['state'][key];b=after['state'][key]
                    assert int(b['step'])==int(a['step'])+(p['updates'] if mutable else 0)
                    if not mutable:equal(a,b,0,0)
                counters[f'{number}/{group["name"]}']=sorted({int(after['state'][key]['step']) for key in group['params']})
        rows[arm]=dict(checkpoint_sha256=sha(directory/'checkpoint.pt'),updates=p['updates'],
            source_model_two_Adam_exact=True,sampler_and_all_RNG_replayed=True,static_geometry_physics_Adam_graphs_frozen=True,
            static_points=len(source['_xyz']),dynamic_points=len(f['model']['pools.0._xyz']),optimizer_steps=counters)
        traces.append(trace)
    assert traces[0]==traces[1]
    assert sha(ROOT/p['parent'])==p['parent_sha256'] and sha(ROOT/p['static_parent'])==p['static_parent_sha256']
    return dict(passed=True,arms=rows,matched_dynamic_start_Adam_RNG_camera_and_budget=True,
                static_source_count_and_Adam_history_differ=True,independent_geometry_GT=False)


if __name__=='__main__':
    from variants.staged_reconstruction.prepare_data import write
    result=audit();write(BASE/'v1/independent_audit.json',result);print('J7_INDEPENDENT_COMPLETE_STATE_AUDIT_PASS',flush=True)
