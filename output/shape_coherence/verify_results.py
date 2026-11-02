"""Independent boundary, snapshot and media verification; does not select or train."""
from pathlib import Path
import sys
import zipfile
import math
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.shape_coherence.run import RUNS,REPORTS,OUT,NAME,CONTROL,assert_frozen
from variants.metric_tuning import run as engine
from output.shape_coherence.continue_run import control_unchanged


def main():
    assert_frozen(ROOT/'variants/shape_coherence/code');control_unchanged()
    d=RUNS/NAME/'rubber_duck';p=d/'checkpoints/resume.pt';receipt=engine.read_json(p.with_suffix('.json'))
    assert receipt['iteration']==50000 and receipt['sha256']==engine.sha256(p)
    branch=engine.read_json(d/'branch.json')
    assert branch['identical_full_checkpoint_sha256']==engine.read_json(REPORTS/'source_freeze.json')['full_3k_sha256']
    assert branch['tensor_optimizer_RNG_bytes_unchanged'] and branch['additional_training_budget']==47000
    assert engine.read_json(d/'metric_config.json')['shape_enabled']
    from variants.c1_flow_ablation.results import motion_module
    m=motion_module()
    from reproduction.scripts.training_checkpoint import load_state
    full=load_state(p)
    def finite(v):
        if torch.is_tensor(v):return bool(v.isfinite().all())
        if isinstance(v,dict):return all(finite(x) for x in v.values())
        if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
        return True
    assert all(finite(full[k]) for k in ['gaussian_parameters','velocity_net','grid4d','deform','gaussian_optimizer','deform_optimizer'])
    g,deform=m.load(d,'50000')
    for key,module in [('grid4d',deform.grid4d),('deform',deform.deform),('velocity_net',g.velocity_net)]:
        assert all(torch.equal(v.cpu(),module.state_dict()[k].cpu()) for k,v in full[key].items())
    assert all(torch.allclose(v.cpu(),getattr(g,k).detach().cpu(),atol=1e-7,rtol=1e-6) for k,v in full['gaussian_parameters'].items())
    del g,deform,full;torch.cuda.empty_cache()
    side=p.with_suffix('.shape.pt');s=engine.read_json(p.with_suffix('.metrics.json'))
    assert s['shape_graph_sha256']==engine.sha256(side)
    graph=torch.load(side,map_location='cpu',weights_only=True)
    assert graph['iteration']==50000 and graph['checkpoint_sha256']==receipt['sha256']
    initial=torch.load(d/'shape_reference/reference_graph.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,graph['graph'][k]) for k,v in initial.items() if torch.is_tensor(v))
    labels=initial['labels'];a=initial['anchors'];n=initial['neighbors']
    assert (labels[a]>=0).all() and (labels[n]==labels[a,None]).all()
    lock=engine.read_json(RUNS/'selection_lock.json')
    assert lock['selection_split']=='validation' and lock['models'][NAME]==engine.read_json(d/'metric_best.json')['PSNR']
    assert lock['models']['C1_repaired']==engine.read_json(CONTROL/'metric_best.json')['PSNR']
    events=[__import__('json').loads(line) for line in (d/'shape_loss.jsonl').read_text().splitlines()]
    assert [e['iteration'] for e in events if e.get('event')=='graph_built']==[16000]
    losses=[e for e in events if 'weighted_loss' in e]
    assert [e['iteration'] for e in losses]==list(range(16100,50001,100))
    assert all(math.isfinite(e['weighted_loss']) for e in losses)
    per=engine.read_json(d/'per_view.json');maximum=0
    for tag in ['best','50000']:
        for i in range(50):
            filename=f'{i:05d}.png';raw=d/'test'/('ours_'+tag)
            with Image.open(raw/'gt'/filename) as im:gt=im.convert('RGB')
            with Image.open(raw/'renders'/filename) as im:render=im.convert('RGB')
            with Image.open(CONTROL/'test'/('ours_'+tag)/'gt'/filename) as im:assert gt.tobytes()==im.convert('RGB').tobytes()
            with Image.open(OUT/NAME/'pairs'/tag/filename) as im:
                assert im.size==(3200,950)
                assert im.crop((0,50,1600,950)).tobytes()==gt.tobytes()
                assert im.crop((1600,50,3200,950)).tobytes()==render.tobytes()
            error=-10*math.log10(max(np.mean((np.asarray(gt,dtype=float)/255-np.asarray(render,dtype=float)/255)**2),1e-12))
            maximum=max(maximum,abs(error-per['ours_'+tag]['PSNR'][filename]))
    assert maximum<1e-4
    import imageio_ffmpeg
    videos=[]
    for tag in ['best','50000']:
        movie=OUT/NAME/('GT_Render_'+tag+'.mp4');frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(movie))
        assert frames==50 and abs(seconds-5)<.11
        videos.append({'tag':tag,'frames':frames,'seconds':seconds,'sha256':engine.sha256(movie)})
    archive=OUT/(NAME+'_GT_Render.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        members=set(z.namelist())
        expected={p.relative_to(OUT/NAME).as_posix() for p in (OUT/NAME).rglob('*') if p.is_file() and not p.name.endswith('.encode.log')}
        assert members==expected
        import hashlib
        for member in members:
            assert hashlib.sha256(z.read(member)).hexdigest()==engine.sha256(OUT/NAME/member)
    snapshots={str(p.relative_to(d)):engine.sha256(p) for tag in ['best','50000'] for folder in ['point_cloud','deform']
               for p in (d/folder/('iteration_'+tag)).rglob('*') if p.is_file()}
    result={'status':'passed','manual_review_pending':True,'historical_control_SHA_unchanged':True,
        'source_data_SHA_verified':True,'full_checkpoint_sha256':receipt['sha256'],'full_snapshot_parity':True,
        'snapshot_sha256':snapshots,'common_full_3k_sha256':branch['identical_full_checkpoint_sha256'],
        'graph_frozen_and_no_cross_label_edges':True,'graph_sha256':engine.sha256(d/'shape_reference/reference_graph.pt'),
        'parameters_and_optimizer_states_finite':True,'selection_lock_sha256':engine.sha256(RUNS/'selection_lock.json'),
        'logged_shape_events':len(losses),'shape_schedule_interval':20,'all_shape_evaluations_individually_logged':False,
        'every_100_steps_logged_only':True,'last_iteration_has_no_optimizer_step_in_original_PIDG':True,
        'full_camera_sampling_log_available':False,'no_actual_schedule_parity_claim':True,
        'pixel_identity':True,'independent_PSNR_max_difference':maximum,'videos':videos,
        'ZIP_all_member_bytes_verified':True,'ZIP_sha256':engine.sha256(archive)}
    engine.write_json(REPORTS/'final_verification.json',result)
    print('BOUNDARIES / WEIGHTS / PIXELS / VIDEOS / ZIP VERIFIED; MANUAL REVIEW PENDING')


if __name__=='__main__':main()
