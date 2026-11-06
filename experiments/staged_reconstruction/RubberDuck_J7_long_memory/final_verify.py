"""Independent CPU final checkpoint, native artifact and decoding audit."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData
import torch
from variants.staged_reconstruction.prepare_data import ROOT,DATA,read,sha,write
from experiments.staged_reconstruction.RubberDuck_J7_long_memory.audit import audit,B,R

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-final-audit',action='store_true')
    if not parser.parse_args().enable_final_audit:raise SystemExit('Read-only audit default off')
    torch.set_num_threads(4);report=B/'v1';p=read(B/'protocol.json');ev=read(report/'evaluation.json');vid=read(report/'videos.json')
    receipts={}
    for receipt in [B/'preregistration.json',report/'source_freeze.json',report/'evaluation_freeze.json']:
        files=read(receipt);files=files.get('files',files)
        for name,h in files.items():assert sha(ROOT/name)==h,name
        receipts[receipt.name]=len(files)
    full=audit();assert full==read(report/'independent_audit.json')==ev['checkpoint_audits']
    load=lambda x:torch.load(x,map_location='cpu',weights_only=False)
    states={'J7_parent':load(ROOT/p['parent'])['model']}
    for arm in p['arms']:
        f=load(R/'formal'/arm/'checkpoint.pt');states[arm]=f['model']
        assert f['protocol_sha256']==sha(B/'protocol.json') and f['source_sha256']==sha(report/'source_freeze.json')
        assert f['parent_sha256']==p['parent_sha256'] and not f['physics_updated']
        assert f['candidate_static_has_no_color_threshold'] and f['dynamic_historical_color_lineage']
    fields=[];world=[]
    for label,item in ev['plys'].items():
        path=ROOT/item['path'];assert sha(path)==item['sha256'];vertex=PlyData.read(path)['vertex']
        assert len(vertex)==item['vertices'] and all(np.isfinite(vertex[k]).all() for k in vertex.data.dtype.names)
        name,kind,*tail=label.split('/')
        if kind=='dynamic_world':
            assert tail[0] in ['2','100','200'] and len(vertex)==p['dynamic_points'];world.append(label);continue
        prefix='static.' if kind=='static' else 'pools.0.';state=states[name]
        for key,props in [('_xyz',['x','y','z']),('_opacity',['opacity']),('_scaling',['scale_0','scale_1','scale_2']),('_rotation',['rot_0','rot_1','rot_2','rot_3'])]:
            assert np.array_equal(np.stack([vertex[k] for k in props],1),state[prefix+key].numpy()),(label,key)
        for key,tag in [('_features_dc','f_dc_'),('_features_rest','f_rest_')]:
            expected=state[prefix+key].transpose(1,2).flatten(start_dim=1).numpy()
            assert np.array_equal(np.stack([vertex[tag+str(i)] for i in range(expected.shape[1])],1),expected),(label,key)
        assert all(np.all(vertex[k]==0) for k in ['nx','ny','nz'])
        assert np.all(vertex['dynamic']==(kind!='static'));fields.append(label)
    assert len(fields)==6 and len(world)==9
    for name,h in ev['diagnostics']['images'].items():
        assert sha(ROOT/name)==h
        with Image.open(ROOT/name) as im:
            if '_native' in name:assert im.size==(1600,900)
            im.verify()
    meta=read(DATA/'transforms_train.json')
    for cid,item in ev['diagnostics']['fixed_cameras'].items():
        frame=next(f for f in meta['frames'] if int(Path(f['file_path']).stem)==int(cid))
        assert item['frame']==frame['file_path'] and item['camera_transform']==frame['transform_matrix']
    rows=ev['validation']['frames'];vm=read(ROOT/'data/derived/kinematic_resplit/transforms_val.json')
    assert len(rows)==30 and [x['frame'] for x in rows]==[f['file_path'] for f in vm['frames']]
    for name in states:
        for key,mean in ev['validation']['means'][name].items():assert abs(float(np.mean([row[name][key] for row in rows]))-mean)<1e-12
    v=ev['validation'];q=v['means']['temporal_memory']['PSNR']
    assert v['PSNR_nonregression_gate']==(q>=v['means']['rgb_control']['PSNR']-.3)
    assert v['parent_nonregression_gate']==(q>=v['means']['J7_parent']['PSNR']-.3)
    decoded={}
    for cid,item in vid['videos'].items():
        path=ROOT/item['path'];assert sha(path)==item['sha256']
        assert sha(ROOT/item['decoded_contact'])==item['decoded_contact_sha256']
        cap=cv2.VideoCapture(str(path));count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(1226,1440);count+=1
        cap.release();assert count==item['decoded_frames']==200
        content=path.read_bytes();assert content.index(b'moov')<content.index(b'mdat');decoded[cid]=count
    controller=read(report/'controller_audit.json');assert controller['passed']
    write(report/'final_validation.json',dict(passed=True,source_receipts=receipts,complete_state=full,
        all4000_samples=True,full_Gaussian_PLY_fields_exact=fields,world_XYZ_evaluator_parity_and_hash_verified=world,
        images_verified=len(ev['diagnostics']['images']),all_native_1600x900=True,
        validation_30_means_and_both_gates_recomputed=True,video_independent_full_decode=decoded,
        faststart_verified=True,controller_audit_sha256=sha(report/'controller_audit.json'),
        independent_geometry_GT=False,browser_playback_not_claimed=True,source_sha256=sha(Path(__file__))))
    print('J8_FINAL_ARTIFACT_AUDIT_PASS',flush=True)

if __name__=='__main__':main()
