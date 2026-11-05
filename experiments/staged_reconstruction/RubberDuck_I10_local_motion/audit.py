"""Independent CPU checkpoint/artifact audit, with all Gaussian PLY fields."""
import argparse
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData
import torch
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha
from variants.staged_reconstruction.joint_rgb.core import assert_equivalent

B=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_local_motion'
R=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_local_motion/v1'


def finite(value):
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite(v)


def run():
    torch.set_num_threads(4);p=read(B/'protocol.json');ev=read(B/'v1/evaluation.json');videos=read(B/'v1/videos.json')
    for receipt in [B/'preregistration.json',B/'v1/source_freeze.json',B/'v1/evaluation_freeze.json']:
        values=read(receipt)
        for name,h in values.get('files',values).items():assert sha(ROOT/name)==h,name
    parent=torch.load(ROOT/p['parent'],map_location='cpu',weights_only=False);states={'J6_parent':parent['model']};initials=[];traces=[]
    assert sha(ROOT/p['parent'])==p['parent_sha256']
    for arm in p['arms']:
        directory=R/'formal'/arm
        for name in ['initial','checkpoint']:
            path=directory/f'{name}.pt';assert sha(path)==read(path.with_suffix('.json'))['sha256']
        initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False)
        final=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert final['iteration']==2000 and final['protocol_sha256']==sha(B/'protocol.json') and final['source_sha256']==sha(B/'v1/source_freeze.json')
        assert final['evidence_sha256']==sha(ROOT/p['evidence']) and not final['color_static_reference'] and final['static_fully_frozen']
        assert not final['old_I10_target_neutralization'] and not final['physics_updated']
        for name in ['model','optimizers','bindings','physics_graphs','rng_python','rng_numpy','rng_torch','rng_cuda','camera_stack']:
            assert_equivalent(initial[name],parent[name],0,0)
        rng=random.Random();rng.setstate(initial['rng_python']);stack=list(initial['camera_stack']);trace=[]
        for _ in range(2000):
            if not stack:stack=rng.sample(range(120),120)
            trace.append(stack.pop())
        assert final['sample_trace']==trace and final['camera_stack']==stack and final['rng_python']==rng.getstate()
        assert initial['bindings']==final['bindings'];assert_equivalent(initial['physics_graphs'],final['physics_graphs'],0,0)
        for name,value in initial['model'].items():
            if name.startswith(('static.','physics.')) or name=='centers':assert torch.equal(value,final['model'][name])
        for i,(before,after) in enumerate(zip(initial['optimizers'],final['optimizers'])):
            assert before['param_groups']==after['param_groups']
            for group in before['param_groups']:
                for key in group['params']:
                    x=before['state'].get(key);y=after['state'].get(key)
                    if i==0 or group['name']=='PIDG_velocity_stress':assert_equivalent(x,y,0,0)
                    else:assert int(y['step'])==int(x['step'])+2000
        finite(final['model']);finite(final['optimizers']);initials.append(initial);traces.append(trace);states[arm]=final['model']
    assert_equivalent(initials[0],initials[1],0,0);assert traces[0]==traces[1]
    fields=[]
    for label,item in ev['plys'].items():
        path=ROOT/item['path'];assert sha(path)==item['sha256'];vertex=PlyData.read(path)['vertex'];assert len(vertex)==item['vertices']
        assert all(np.isfinite(vertex[k]).all() for k in vertex.data.dtype.names)
        model,kind,*_=label.split('/')
        if kind=='dynamic_world':continue
        prefix='static.' if kind=='static' else 'pools.0.';state=states[model]
        for name,props in [('_xyz',['x','y','z']),('_opacity',['opacity']),('_scaling',['scale_0','scale_1','scale_2']),('_rotation',['rot_0','rot_1','rot_2','rot_3'])]:
            assert np.array_equal(np.stack([vertex[k] for k in props],1),state[prefix+name].numpy()),(label,name)
        for field,tag in [('_features_dc','f_dc_'),('_features_rest','f_rest_')]:
            expected=state[prefix+field].transpose(1,2).flatten(start_dim=1).numpy()
            assert np.array_equal(np.stack([vertex[tag+str(i)] for i in range(expected.shape[1])],1),expected),(label,field)
        assert all(np.all(vertex[k]==0) for k in ['nx','ny','nz']) and np.all(vertex['dynamic']==(kind!='static'));fields.append(label)
    for name,h in ev['diagnostics']['images'].items():
        assert sha(ROOT/name)==h
        with Image.open(ROOT/name) as im:im.verify()
    counts=[]
    for item in videos['videos'].values():
        path=ROOT/item['path'];assert sha(path)==item['sha256']
        cap=cv2.VideoCapture(str(path));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==200;counts.append(count)
        assert sha(ROOT/item['decoded_contact'])==item['decoded_contact_sha256']
    rows=ev['validation']['frames'];assert len(rows)==30
    for model in states:
        for key,value in ev['validation']['means'][model].items():assert abs(np.mean([r[model][key] for r in rows])-value)<1e-12
    write(B/'v1/independent_audit.json',dict(passed=True,source_protocol_input_SHA=True,
        model_twoAdam_RNG_sampler_initial_parent_exact=True,all4000_samples_replayed=True,
        static_model_Adam_physics_model_Adam_graphs_exact=True,PLY_all_Gaussian_fields_exact=fields,
        world_XYZ_checked_by_native_evaluator=True,images_verified=len(ev['diagnostics']['images']),
        two_H264_independently_decoded=counts,all30_RGB_averages_recomputed=True,
        independent_geometry_GT=False,training_tracks_not_independent_validation=True,source_sha256=sha(Path(__file__))))
    print('L1_INDEPENDENT_ARTIFACT_AUDIT_PASS',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--enable-audit',action='store_true')
    if not parser.parse_args().enable_audit:raise SystemExit('Audit default off')
    run()
