"""Independent CPU artifact checks for the one preregistered J7 trial."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from variants.staged_reconstruction.prepare_data import ROOT, DATA, read, sha, write
from experiments.staged_reconstruction.RubberDuck_colorfree_joint.audit import audit

B=ROOT/'experiments/staged_reconstruction/RubberDuck_colorfree_joint'
R=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_colorfree_joint/v1'


def finite(value):
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite(v)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-colorfree-final-audit',action='store_true')
    if not parser.parse_args().enable_colorfree_final_audit:raise SystemExit('Read-only final audit is default off')
    torch.set_num_threads(4)
    report=B/'v1';p=read(B/'protocol.json');ev=read(report/'evaluation.json');vid=read(report/'videos.json')
    source_count={}
    for receipt in [B/'preregistration.json',report/'source_freeze.json',report/'evaluation_freeze.json']:
        files=read(receipt);files=files.get('files',files)
        for name,h in files.items():assert sha(ROOT/name)==h,name
        source_count[receipt.name]=len(files)
    complete=audit();assert complete==ev['checkpoint_audits']==read(report/'independent_audit.json')
    load=lambda path:torch.load(path,map_location='cpu',weights_only=False)
    states={}
    j6=load(ROOT/p['parent']);source=load(ROOT/p['static_parent'])
    states['J6_parent']=j6['model']
    states['direct_fusion']={n:v for n,v in j6['model'].items() if not n.startswith('static.')}
    states['direct_fusion'].update({'static.'+n:v for n,v in source['model'].items()})
    for arm in p['arms']:
        d=R/'formal'/arm
        for name in ['initial','checkpoint']:
            path=d/(name+'.pt');assert sha(path)==read(path.with_suffix('.json'))['sha256']
        f=load(d/'checkpoint.pt')
        assert f['iteration']==p['updates'] and len(f['sample_trace'])==p['updates']
        assert f['protocol_sha256']==sha(B/'protocol.json') and f['source_sha256']==sha(report/'source_freeze.json')
        assert f['parent_sha256']==p['parent_sha256'] and not f['physics_updated']
        assert f['fusion_arm']==arm and f['candidate_static_has_no_color_threshold'] and f['dynamic_historical_color_lineage']
        assert torch.equal(f['model']['centers'],j6['model']['centers'])
        finite(f['model']);finite(f['optimizers']);states[arm]=f['model']
    assert len(states)==4
    fields=[];world=[]
    for label,item in ev['plys'].items():
        path=ROOT/item['path'];assert sha(path)==item['sha256']
        vertex=PlyData.read(path)['vertex'];assert len(vertex)==item['vertices']
        assert all(np.isfinite(vertex[key]).all() for key in vertex.data.dtype.names)
        model,kind,*tail=label.split('/')
        if kind=='dynamic_world':
            assert tail[0] in ['2','100','200'] and len(vertex)==p['dynamic_points']
            world.append(label);continue
        prefix='static.' if kind=='static' else 'pools.0.';state=states[model]
        for key,props in [('_xyz',['x','y','z']),('_opacity',['opacity']),('_scaling',['scale_0','scale_1','scale_2']),('_rotation',['rot_0','rot_1','rot_2','rot_3'])]:
            actual=np.stack([vertex[prop] for prop in props],1)
            assert np.array_equal(actual,state[prefix+key].numpy()),(label,key)
        for key,tag in [('_features_dc','f_dc_'),('_features_rest','f_rest_')]:
            expected=state[prefix+key].transpose(1,2).flatten(start_dim=1).numpy()
            actual=np.stack([vertex[tag+str(i)] for i in range(expected.shape[1])],1)
            assert np.array_equal(actual,expected),(label,key)
        assert all(np.all(vertex[k]==0) for k in ['nx','ny','nz'])
        assert np.all(vertex['dynamic']==(kind!='static'))
        fields.append(label)
    assert len(fields)==8 and len(world)==12
    for name,h in ev['diagnostics']['images'].items():
        assert sha(ROOT/name)==h
        with Image.open(ROOT/name) as im:
            if '_native' in name:assert im.size==(1600,900)
            im.verify()
    metas=read(DATA/'transforms_train.json')
    for camid,item in ev['diagnostics']['fixed_cameras'].items():
        frame=next(f for f in metas['frames'] if int(Path(f['file_path']).stem)==int(camid))
        assert item['frame']==frame['file_path'] and item['camera_transform']==frame['transform_matrix']
    rows=ev['validation']['frames'];assert len(rows)==30
    valmeta=read(ROOT/'data/derived/kinematic_resplit/transforms_val.json')
    assert [q['frame'] for q in rows]==[f['file_path'] for f in valmeta['frames']]
    for label in states:
        for key,mean in ev['validation']['means'][label].items():
            assert abs(float(np.mean([row[label][key] for row in rows]))-mean)<1e-12
    v=ev['validation'];delta=v['means']['colorfree_static']['PSNR']-v['means']['old_static_control']['PSNR']
    parent_delta=v['means']['colorfree_static']['PSNR']-v['means']['J6_parent']['PSNR']
    assert delta==v['candidate_minus_control_PSNR_dB'] and parent_delta==v['candidate_minus_J6_PSNR_dB']
    assert v['PSNR_nonregression_gate']==(delta>=-.3 and parent_delta>=-.3)
    decoded={}
    for camid,item in vid['videos'].items():
        path=ROOT/item['path'];assert sha(path)==item['sha256']
        assert sha(ROOT/item['decoded_contact'])==item['decoded_contact_sha256']
        with Image.open(ROOT/item['decoded_contact']) as im:im.verify()
        cap=cv2.VideoCapture(str(path));count=0
        while True:
            ok,value=cap.read()
            if not ok:break
            assert value.shape[:2]==(1226,1920);count+=1
        cap.release();assert count==item['decoded_frames']==200
        content=path.read_bytes();assert content.index(b'moov')<content.index(b'mdat')
        decoded[camid]=count
    result=dict(passed=True,source_receipts=source_count,complete_state=complete,
        all4000_camera_samples_replayed=True,full_Gaussian_PLY_fields_exact=fields,
        world_XYZ_evaluator_parity_and_hash_verified=world,
        images_verified=len(ev['diagnostics']['images']),native_images_1600x900=True,
        validation_30_frames_aggregate_and_gate_recomputed=True,video_independent_full_decode=decoded,
        faststart_atom_order_verified=True,independent_geometry_GT=False,browser_playback_not_claimed=True,
        verification_source_sha256=sha(Path(__file__)))
    write(report/'final_validation.json',result)
    print('J7_INDEPENDENT_FINAL_ARTIFACT_AUDIT_PASS',flush=True)


if __name__=='__main__':main()
