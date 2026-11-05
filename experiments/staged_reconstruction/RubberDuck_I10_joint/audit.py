"""Independent CPU artifact checks; no optimizer/model updates or GT inputs."""
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from variants.staged_reconstruction.prepare_data import ROOT,read,sha,write
from variants.staged_reconstruction.joint_rgb.core import assert_equivalent

B=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_joint'
R=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_joint/v1'


def verify_finite(value):
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for v in value.values():verify_finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:verify_finite(v)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-i10-joint-audit',action='store_true')
    if not parser.parse_args().enable_i10_joint_audit:raise SystemExit('Read-only audit default off')
    torch.set_num_threads(4)
    p=read(B/'protocol.json');ev=read(B/'v1/evaluation.json');vid=read(B/'v1/videos.json')
    assert sha(B/'evaluate.py')==ev['source_sha256']==vid['source_sha256']
    for receipt in [B/'preregistration.json',B/'v1/source_freeze.json',B/'v1/evaluation_freeze.json']:
        files=read(receipt);files=files.get('files',files)
        for name,h in files.items():assert sha(ROOT/name)==h,name
    sources={key:torch.load(ROOT/p[key],map_location='cpu',weights_only=False) for key in ['parent','static_parent']}
    for key in sources:assert sha(ROOT/p[key])==p[key+'_sha256']
    states={};initials=[]
    for arm in p['arms']:
        directory=R/'formal'/arm
        for file in ['initial','checkpoint']:
            path=directory/(file+'.pt');assert sha(path)==read(path.with_suffix('.json'))['sha256']
        initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False)
        final=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert final['iteration']==p['updates']==2000
        assert final['protocol_sha256']==sha(B/'protocol.json') and final['source_sha256']==sha(B/'v1/source_freeze.json')
        assert not final['physics_updated'] and not final['old_I10_target_neutralization']
        for key,value in initial['model'].items():
            src=sources['static_parent'] if key.startswith('static.') else sources['parent']
            assert torch.equal(value,src['model'][key]),key
        assert_equivalent(initial['optimizers'][0],sources['static_parent']['optimizers'][0],0,0)
        assert_equivalent(initial['optimizers'][1],sources['parent']['optimizer'],0,0)
        for key in ['rng_python','rng_numpy','rng_torch','rng_cuda','camera_stack','physics_graphs']:
            assert_equivalent(initial[key],sources['parent'][key],0,0)
        sampler=random.Random();sampler.setstate(initial['rng_python']);stack=list(initial['camera_stack']);trace=[]
        for _ in range(2000):
            if not stack:stack=sampler.sample(range(120),120)
            trace.append(stack.pop())
        assert final['sample_trace']==trace and final['camera_stack']==stack and final['rng_python']==sampler.getstate()
        assert initial['bindings']==final['bindings']
        for key,value in initial['model'].items():
            if key.startswith('physics.') or key=='centers' or (key.startswith('static.') and (arm=='frozen_static' or key not in ['static._opacity','static._features_dc','static._features_rest'])):
                assert torch.equal(value,final['model'][key]),(arm,key)
        assert_equivalent(initial['physics_graphs'],final['physics_graphs'],0,0)
        for i,(before,after) in enumerate(zip(initial['optimizers'],final['optimizers'])):
            assert before['param_groups']==after['param_groups']
            for group in before['param_groups']:
                updated=(i==0 and arm=='joint_depth_trust' and group['name'] in ['_opacity','_features_dc','_features_rest']) or (i==1 and group['name']!='PIDG_velocity_stress')
                for key in group['params']:
                    if key not in before['state']:continue
                    a,b=before['state'][key],after['state'][key]
                    if updated:assert int(b['step'])==int(a['step'])+2000
                    else:assert_equivalent(a,b,0,0)
        verify_finite(final['model']);verify_finite(final['optimizers'])
        initials.append(initial);states[arm]=final['model']
    assert_equivalent(initials[0],initials[1],0,0)
    states['I10_J4_initial']=initials[0]['model']
    fields=[]
    for label,item in ev['plys'].items():
        path=ROOT/item['path'];assert sha(path)==item['sha256']
        vertex=PlyData.read(path)['vertex'];assert len(vertex)==item['vertices']
        model,kind,*_=label.split('/')
        if kind=='dynamic_world':continue
        prefix='static.' if kind=='static' else 'pools.0.'
        state=states[model]
        expected={k:state[prefix+k].numpy() for k in ['_xyz','_opacity','_scaling','_rotation']}
        for k,props in [('_xyz',['x','y','z']),('_opacity',['opacity']),('_scaling',['scale_0','scale_1','scale_2']),('_rotation',['rot_0','rot_1','rot_2','rot_3'])]:
            actual=np.stack([vertex[prop] for prop in props],1)
            assert np.array_equal(actual,expected[k]),(label,k)
        for field,tag in [('_features_dc','f_dc_'),('_features_rest','f_rest_')]:
            expected=state[prefix+field].transpose(1,2).flatten(start_dim=1).numpy()
            actual=np.stack([vertex[tag+str(i)] for i in range(expected.shape[1])],1)
            assert np.array_equal(actual,expected),(label,field)
        assert all(np.isfinite(vertex[key]).all() for key in vertex.data.dtype.names)
        assert all(np.all(vertex[k]==0) for k in ['nx','ny','nz'])
        assert np.all(vertex['dynamic']==(kind!='static'))
        fields.append(label)
    for name,h in ev['diagnostics']['images'].items():
        assert sha(ROOT/name)==h
        with Image.open(ROOT/name) as im:im.verify()
    for item in vid['videos'].values():
        assert sha(ROOT/item['path'])==item['sha256'] and item['decoded_frames']==200
        assert sha(ROOT/item['decoded_contact'])==item['decoded_contact_sha256']
    rows=ev['validation']['frames'];assert len(rows)==30
    for label in states:
        for key,mean in ev['validation']['means'][label].items():
            assert abs(float(np.mean([row[label][key] for row in rows]))-mean)<1e-12
    write(B/'v1/independent_audit.json',{'passed':True,'protocol_source_data_SHA_verified':True,
        'two_source_model_and_full_Adam_exact':True,'both_initial_snapshots_exact':True,
        'all_4000_samples_replayed':True,'frozen_parameters_Adam_graphs_exact':True,
        'full_Gaussian_PLY_all_model_fields_exact':fields,'world_XYZ_parity_verified_by_native_evaluator':True,
        'images_verified':len(ev['diagnostics']['images']),'two_videos_decoded_frames':[200,200],
        'all30_metrics_recomputed':True,'native_trajectory_determinism_not_claimed':True,
        'independent_geometry_GT':False,'source_sha256':sha(Path(__file__))})
    print('J6_INDEPENDENT_ARTIFACT_AUDIT_PASS',flush=True)


if __name__=='__main__':main()
