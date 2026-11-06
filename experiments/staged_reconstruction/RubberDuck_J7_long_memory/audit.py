"""Independent full-state and all-camera replay of the matched J8 pair."""
import random
import torch
from variants.staged_reconstruction.prepare_data import ROOT,read,sha
from variants.staged_reconstruction.joint_rgb.core import assert_equivalent as equal

B=ROOT/'experiments/staged_reconstruction/RubberDuck_J7_long_memory'
R=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_J7_long_memory/v1'

def audit():
    p=read(B/'protocol.json');load=lambda x:torch.load(x,map_location='cpu',weights_only=False)
    parent=load(ROOT/p['parent']);rows={};initials=[];traces=[]
    assert sha(ROOT/p['parent'])==p['parent_sha256']
    for arm in p['arms']:
        d=R/'formal'/arm;i=load(d/'initial.pt');f=load(d/'checkpoint.pt')
        for tag in ['initial','checkpoint']:assert sha(d/(tag+'.pt'))==read(d/(tag+'.json'))['sha256']
        assert i['iteration']==0 and f['iteration']==p['updates']==len(f['sample_trace'])
        for key in ['model','optimizers','bindings','physics_graphs','rng_python','rng_numpy','rng_torch','rng_cuda','camera_stack']:
            equal(i[key],parent[key],0,0)
        for name,value in i['model'].items():
            if name.startswith('physics.') or name=='centers' or (name.startswith('static.') and name[7:] not in p['static_trainable']):
                assert torch.equal(value,f['model'][name]),(arm,name)
            assert torch.isfinite(f['model'][name]).all(),(arm,name)
        equal(i['physics_graphs'],f['physics_graphs'],0,0)
        for key in ['rng_numpy','rng_torch','rng_cuda']:equal(i[key],f[key],0,0)
        sampler=random.Random();sampler.setstate(parent['rng_python']);stack=list(parent['camera_stack']);trace=[]
        for _ in range(p['updates']):
            if not stack:stack=sampler.sample(range(120),120)
            trace.append(stack.pop())
        assert trace==f['sample_trace'] and stack==f['camera_stack'] and sampler.getstate()==f['rng_python']
        assert i['bindings']==f['bindings'] and f['controller']['calls']==2000
        assert f['self_depth_teacher_parent_sha256']==p['parent_sha256']
        counters={}
        for number,(before,after) in enumerate(zip(i['optimizers'],f['optimizers'])):
            assert before['param_groups']==after['param_groups']
            for group in after['param_groups']:
                mutable=(group['name'] in p['static_trainable']) if number==0 else group['name']!='PIDG_velocity_stress'
                for key in group['params']:
                    x,y=before['state'][key],after['state'][key]
                    assert int(y['step'])==int(x['step'])+(2000 if mutable else 0)
                    if not mutable:equal(x,y,0,0)
                    assert all(not isinstance(v,torch.Tensor) or torch.isfinite(v).all() for v in y.values())
                counters[f'{number}/{group["name"]}']=sorted({int(after['state'][k]['step']) for k in group['params']})
        rows[arm]=dict(checkpoint_sha256=sha(d/'checkpoint.pt'),source_model_two_Adam_exact=True,
            all2000_samples_and_RNG_replayed=True,static_geometry_physics_Adam_graphs_frozen=True,
            static_points=len(f['model']['static._xyz']),dynamic_points=len(f['model']['pools.0._xyz']),optimizer_steps=counters)
        initials.append(i);traces.append(trace)
    equal(initials[0],initials[1],0,0);assert traces[0]==traces[1]
    return dict(passed=True,arms=rows,matched_complete_J7_start=True,independent_geometry_GT=False)

if __name__=='__main__':
    from variants.staged_reconstruction.prepare_data import write
    write(B/'v1/independent_audit.json',audit());print('J8_FULL_STATE_AUDIT_PASS',flush=True)
