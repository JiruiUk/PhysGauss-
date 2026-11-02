"""Independent CPU snapshot/SHA/pixel/metric/video receipts and review sheets."""
import json
from pathlib import Path
import sys
import zipfile
from datetime import datetime,timezone
import numpy as np
from PIL import Image
import torch
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.momentum_guard.run import NAME,RUNS,REPORTS,assert_frozen
from variants.metric_tuning.run import sha256 as sha,read_json as read,write_json as write
from variants.unified_flow.prepare_data import prepare
OUT=Path(__file__).parent;torch.set_num_threads(4)
assert read(REPORTS/'progress.json')['status']=='completed'
assert_frozen(ROOT/'variants/momentum_guard/code');prepare()
run=RUNS/NAME/'rubber_duck';control=ROOT/'runs/pidg_unified_flow/L1_shared_stress/rubber_duck'
a,b=[read(p/'metric_config.json') for p in [control,run]]
assert {k for k in set(a)|set(b) if a.get(k)!=b.get(k)}=={'momentum_gradient_guard','momentum_shared_ratio'}
assert (run/'model_config.py').read_bytes()==(control/'model_config.py').read_bytes()
branch=read(run/'branch.json');prefix_path=ROOT/'runs/pidg_unified_flow/shared_prefix/rubber_duck/checkpoints/resume.pt'
assert branch['identical_full_checkpoint_sha256']==sha(prefix_path)==read(control/'branch.json')['identical_full_checkpoint_sha256']
file=run/'checkpoints/resume.pt';meta=read(file.with_suffix('.json'))
assert meta['iteration']==50000 and sha(file)==meta['sha256']
state=torch.load(file,map_location='cpu',weights_only=False)
prefix=torch.load(prefix_path,map_location='cpu',weights_only=False)
assert all(v.isfinite().all() for k in ['deform','grid4d','velocity_net','gaussian_parameters'] for v in state[k].values())
assert all(torch.equal(v,state['velocity_net'][k]) for k,v in prefix['velocity_net'].items())
lock=read(RUNS/'selection_lock.json');selected=lock['models'][NAME]
history=[json.loads(s) for s in (run/'metric_history.jsonl').read_text().splitlines()]
expected=max(history,key=lambda r:r['metrics']['PSNR'])
assert len(history)==51 and len({r['iteration'] for r in history})==51
assert selected['iteration']==expected['iteration'] and abs(selected['value']-expected['metrics']['PSNR'])<1e-10
assert (run/'test/ours_best/renders/00000.png').stat().st_mtime>datetime.fromisoformat(lock['locked_utc']).timestamp()
snapshots={}
for tag in ['best','50000']:
    folder=run/'deform'/('iteration_'+tag)
    grid,deform=torch.load(folder/'deform.pth',map_location='cpu',weights_only=False)
    material=run/'point_cloud'/('iteration_'+tag)/'material.pt'
    mat=torch.load(material,map_location='cpu',weights_only=True)
    assert 'shared_stress.weight' in deform and all(v.isfinite().all() for d in [grid,deform] for v in d.values())
    if tag=='50000':
        assert all(torch.equal(v,state[key][k]) for key,d in [('grid4d',grid),('deform',deform)] for k,v in d.items())
        assert all(torch.equal(v,state['velocity_net'][k]) for k,v in mat['velocity_net'].items())
    else:assert str(selected['iteration']) in (folder/'iter.txt').read_text()
    snapshots[tag]={'selected_iteration':selected['iteration'] if tag=='best' else 50000,
        'sha256':{str(p.relative_to(ROOT)):sha(p) for p in [folder/'deform.pth',material,material.with_name('point_cloud.ply')]}}
def rows(folder,name):return [json.loads(s) for s in (folder/name).read_text().splitlines()]
flow=rows(run,'observed_flow.jsonl');momentum=rows(run,'material_momentum.jsonl');guards=rows(run,'momentum_gradient_guard.jsonl')
assert [r['iteration'] for r in momentum]==list(range(8100,50001,100))==[r['iteration'] for r in guards]
assert len(flow)==468 and all(0<r['valid_fraction']<=1 for r in momentum)
same_schedule=[(r['iteration'],r['source'],r['target']) for r in flow]==[(r['iteration'],r['source'],r['target']) for r in rows(control,'observed_flow.jsonl')]
for r in guards:
    assert r['private_gradients_preserved'] and 0<=r['scale']<=1
    for v in r['groups'].values():assert v['guarded_momentum_norm']<=.1*v['image_norm']*(1+1e-5)
out=OUT/NAME;per=read(run/'per_view.json');errors=[];videos=[]
for tag in ['best','50000']:
    raw=run/'test'/('ours_'+tag);assert len(list((out/'pairs'/tag).glob('*.png')))==50
    for i in range(50):
        n=f'{i:05d}.png'
        with Image.open(raw/'gt'/n) as a,Image.open(raw/'renders'/n) as b,Image.open(out/'pairs'/tag/n) as c,Image.open(control/'test/ours_best/gt'/n) as old:
            gt,r,j,ref=[np.asarray(im.convert('RGB')) for im in [a,b,c,old]]
        assert j.shape==(950,3200,3) and np.array_equal(gt,ref)
        assert np.array_equal(gt,j[50:950,:1600]) and np.array_equal(r,j[50:950,1600:])
        psnr=-10*np.log10(np.mean(((gt.astype(np.float64)-r)/255.)**2))
        errors.append(abs(psnr-per['ours_'+tag]['PSNR'][n]))
    assert max(errors)<1e-4
    video=out/f'GT_Render_{tag}.mp4';count,seconds=imageio_ffmpeg.count_frames_and_secs(str(video))
    assert count==50 and abs(seconds-5)<.11
    videos.append({'path':str(video),'frames':count,'seconds':seconds,'sha256':sha(video)})
archive=OUT/(NAME+'_GT_Render.zip')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and sum(n.endswith('.png') for n in z.namelist())==100
ids=[0,21,25,49];png=Image.new('RGB',(1600,1000),'white');mov=png.copy()
for row,dest in enumerate([ROOT/'output/unified_flow/L1_shared_stress',out]):
    for col,i in enumerate(ids):
        with Image.open(dest/'pairs/best'/f'{i:05d}.png') as im:png.paste(im.resize((800,238)),((col%2)*800,row*500+(col//2)*250))
    reader=imageio_ffmpeg.read_frames(str(dest/'GT_Render_best.mp4'),pix_fmt='rgb24');vmeta=next(reader);w,h=vmeta['size']
    assert (w,h)==(3200,950)
    for i,data in enumerate(reader):
        if i in ids:
            col=ids.index(i);im=Image.frombytes('RGB',(w,h),data)
            mov.paste(im.resize((800,238)),((col%2)*800,row*500+(col//2)*250))
png.save(OUT/'png_review.png');mov.save(OUT/'decoded_video_review.png')
fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
for folder,label in [(control,'L1 control'),(run,'M1 gradient guard')]:
    hist=rows(folder,'metric_history.jsonl');axes[0].plot([r['iteration'] for r in hist],[r['metrics']['PSNR'] for r in hist],label=label)
axes[0].set(xlabel='Total iterations',ylabel='Validation PSNR (dB)');axes[0].legend()
motion=read(REPORTS/'motion_validation.json')['results'][0]['metrics']
prior=next(r['metrics'] for r in read(ROOT/'experiments/unified_flow/motion_validation.json')['results'] if r['model']=='L1_shared_stress' and r['snapshot']=='best')
for offset,key,label in [(-.15,'geometric_EPE_px','Trajectory'),(.15,'velocity_EPE_px','Velocity advection')]:
    axes[1].bar(np.arange(2)+offset,[prior[key],motion[key]],width=.29,label=label)
axes[1].set(xticks=[0,1],xticklabels=['L1','M1'],ylabel='Independent estimated 2D EPE (px)');axes[1].legend()
fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
frames=read(ROOT/'data/derived/kinematic_resplit/final_test/transforms_test.json')['frames'];fids=[int(Path(f['file_path']).stem) for f in frames]
assert fids[21]==85
receipt={'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat(),'source_data_SHA_verified':True,
    'full_checkpoint_sha256':meta['sha256'],'common_extended3k_sha256':sha(prefix_path),'no_new_parameters':True,
    'config_difference':['momentum_gradient_guard','momentum_shared_ratio'],'snapshots':snapshots,
    'validation_selection_verified':True,'test_after_selection_lock':True,'unused_physics_unchanged':True,
    'same_observed_flow_frame_schedule_as_control':same_schedule,'flow_events':len(flow),'momentum_events':len(momentum),
    'momentum_optimizer_updates':sum(r['iteration']<50000 for r in guards),'all_gradient_caps_verified':True,
    'pixel_identity_pairs_verified':100,'independent_PSNR_max_error':max(errors),'ZIP_crc_passed':True,
    'ZIP_sha256':sha(archive),'videos':videos,'manual_review_pending':True,'review_frame_ids':[fids[i] for i in ids],
    'analysis_scripts_sha256':{p.name:sha(p) for p in OUT.glob('*.py')}}
write(REPORTS/'final_verification.json',receipt);print(json.dumps(receipt,indent=2))
