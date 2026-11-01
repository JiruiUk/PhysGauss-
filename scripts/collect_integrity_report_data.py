"""Read-only audit of completed50k models for the Chinese technical report."""
import hashlib
import json
import math
from pathlib import Path
import runpy
import numpy as np
from PIL import Image
from plyfile import PlyData
ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'output/pdf/integrity_analysis_assets'
LABELS={'C1_reused':'C1','H1_balanced_coverage':'H1','H2_temporal_cohesion':'H2','H3_combined':'H3',
    'H4_color_density':'H4','F1_feedback':'F1','F2_conservative_retry':'F2'}
SOURCES={}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def track(path):
    path=Path(path);SOURCES[str(path.relative_to(ROOT))]=digest(path);return path


def read(path):return json.loads(track(path).read_text(encoding='utf-8-sig'))


def lines(path):return [json.loads(s) for s in track(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def collect():
    ASSETS.mkdir(parents=True,exist_ok=True)
    status=read(ROOT/'runs/pidg_integrity_continuation/controller.json')
    if status['status']!='completed':raise ValueError('Full continuation/export is not completed')
    assessment=read(ROOT/'experiments/integrity_continuation/assessment.json')
    metadata=read(ROOT/'data/derived/kinematic_resplit/final_test/transforms_test.json')
    frame_ids=[int(Path(f['file_path']).name) for f in metadata['frames']]
    assert frame_ids==list(range(1,201,4))
    train_meta=read(ROOT/'data/derived/kinematic_resplit/training/transforms_train.json')
    val_meta=read(ROOT/'data/derived/kinematic_resplit/training/transforms_test.json')
    train_ids={int(Path(f['file_path']).name) for f in train_meta['frames']}
    val_ids={int(Path(f['file_path']).name) for f in val_meta['frames']}
    test_ids=set(frame_ids)
    assert (len(train_ids),len(val_ids),len(test_ids))==(120,30,50)
    assert not(train_ids&val_ids or train_ids&test_ids or val_ids&test_ids)
    assert train_ids|val_ids|test_ids==set(range(1,201))
    models=[];shared=[]
    for row in assessment['models']:
        d=Path(row['run_absolute']);name=LABELS[row['model']]
        manifest=read(d/'checkpoints/resume.json');assert manifest['iteration']==50000
        assert digest(d/'checkpoints/resume.pt')==manifest['sha256']
        config=read(d/'metric_config.json');settings=runpy.run_path(str(track(d/'model_config.py')))
        assert settings['network_args']['width']==256 and settings['network_args']['depth']==0
        result=read(d/'results.json');per=read(d/'per_view.json')['ours_50000']
        for key in ('PSNR','SSIM','LPIPS','MS-SSIM'):
            assert set(per[key])=={f'{i:05d}.png' for i in range(50)}
            assert abs(math.fsum(per[key].values())/50-row['test_metrics'][key])<1e-5
        branch=read(d/'branch.json');shared.append(branch['identical_full_checkpoint_sha256'])
        material=PlyData.read(str(track(d/'point_cloud/iteration_50000/point_cloud.ply')))['vertex'].data
        scale=np.exp(np.stack([material['scale_'+str(i)] for i in range(3)],-1))
        opacity=1/(1+np.exp(-material['opacity'].astype(np.float64)))
        events=lines(d/'coverage_density.jsonl') if (d/'coverage_density.jsonl').exists() else []
        history=lines(d/'metric_history.jsonl');best=read(d/'metric_best.json')
        assert all(v['selection_split']=='validation' for v in best.values())
        snapshots=[]
        for tag in ['best','integrity_best']:
            if 'ours_'+tag in result:
                source=best['PSNR']['iteration'] if tag=='best' else read(d/'integrity_best.json')['iteration']
                snapshots.append({'tag':tag,'source_iteration':source,'test_metrics':result['ours_'+tag]})
        models.append({'id':name,'name':row['model'],'directory':str(d),'config':config,'settings':{k:settings[k] for k in ('grid_args','network_args')},
            'test':row['test_metrics'],'integrity':row['test_integrity'],'validation':row['validation'],
            'per_view':per,'history':history,'points':len(material),'density_events':events,
            'density_added':sum(e['added'] for e in events),'canonical_opacity_quantiles':np.quantile(opacity,[.1,.5,.9]).tolist(),
            'canonical_scale_quantiles':np.quantile(scale,[.1,.5,.9]).tolist(),'selected_snapshots':snapshots})
    assert len(set(shared))==1
    models.sort(key=lambda m:['C1','H1','H2','H3','H4','F1','F2'].index(m['id']))
    for model in models:model['photometric_per_frame']=[]
    for index,frame in enumerate(metadata['frames']):
        file=f'{index:05d}.png';source=ROOT/'data/derived/kinematic_resplit/final_test'/(frame['file_path']+'.png')
        with Image.open(track(source)) as im:mask=np.asarray(im.getchannel('A').resize((1600,900),Image.Resampling.BICUBIC),dtype=np.float32)/255.>.5
        gt_hashes=[]
        for model in models:
            raw=Path(model['directory'])/'test/ours_50000'
            gt_path=track(raw/'gt'/file);gt_hashes.append(SOURCES[str(gt_path.relative_to(ROOT))])
            with Image.open(gt_path) as im:gt=np.asarray(im.convert('RGB'),dtype=np.float32)/255.
            with Image.open(track(raw/'renders'/file)) as im:rgb=np.asarray(im.convert('RGB'),dtype=np.float32)/255.
            with Image.open(track(raw/'alpha'/file)) as im:a=np.asarray(im.convert('L'),dtype=np.float32)/255.
            error=np.square(rgb-gt).mean(-1)
            fg=float(error[mask].mean());bg=float(error[~mask].mean())
            model['photometric_per_frame'].append({'frame_id':frame_ids[index],'foreground_PSNR':-10*math.log10(max(fg,1e-12)),
                'foreground_mse':fg,'background_mse':bg,'background_contribution_to_full_mse':float(error[~mask].sum()/error.size),
                'foreground_contribution_to_full_mse':float(error[mask].sum()/error.size),'foreground_alpha_mean':float(a[mask].mean()),
                'foreground_alpha_075_080':float(((a>=.75)&(a<.8))[mask].mean()),'foreground_alpha_080_085':float(((a>=.8)&(a<.85))[mask].mean())})
        assert len(set(gt_hashes))==1
    baseline=models[0];pairwise={}
    for model in models:
        photo=model['photometric_per_frame'];model['photometric_mean']={k:float(np.mean([r[k] for r in photo])) for k in photo[0] if k!='frame_id'}
        delta=np.array([model['per_view']['PSNR'][f'{i:05d}.png']-baseline['per_view']['PSNR'][f'{i:05d}.png'] for i in range(50)])
        rows=[{'index':i,'frame_id':frame_ids[i],'PSNR_delta':float(v)} for i,v in enumerate(delta)]
        pairwise[model['id']]={'mean':float(delta.mean()),'median':float(np.median(delta)),'improved':int((delta>0).sum()),'worse':int((delta<0).sum()),
            'minimum':min(rows,key=lambda r:r['PSNR_delta']),'maximum':max(rows,key=lambda r:r['PSNR_delta']),
            'thirds':[{'frame_range':[frame_ids[s.start],frame_ids[s.stop-1]],'mean_delta':float(delta[s].mean()),'improved':int((delta[s]>0).sum()),'count':len(delta[s])} for s in [slice(0,17),slice(17,34),slice(34,50)]],
            'per_frame':rows}
    object_id=[]
    identity=read(ROOT/'experiments/object_id/identity_exports.json')
    for name in ['O1_frozen_ID','O2_selfsup_ID']:
        d=ROOT/'runs/pidg_object_id'/name/'rubber_duck'
        assert read(d/'branch.json')['identical_full_checkpoint_sha256']==shared[0]
        config=read(d/'metric_config.json');scores=read(d/'results.json')['ours_50000']
        diagnostic=next(x for x in identity['summaries'] if x['model']==name and x['snapshot']=='50000')
        object_id.append({'id':name[:2],'name':name,'test':scores,'config':config,'diagnostic':diagnostic,
            'validation':lines(d/'metric_history.jsonl')[-1]['metrics']})
    for path in ['variants/integrity_tuning/runtime.py','variants/integrity_tuning/model.py','variants/integrity_feedback/runtime.py',
        'variants/integrity_feedback/model.py','variants/color_repair/runtime.py','variants/color_repair/model.py','variants/color_repair/color.py',
        'variants/color_repair/code/scene_PIDG/network.py','variants/object_id/model.py','variants/object_id/runtime.py']:
        track(ROOT/path)
    context=[]
    for family,names in {
        'pidg_color_repair':['C2_repaired','C3_repaired'],
        'pidg_c1_material':['P0_C1_control','P1_fixed_E','P2_learn_E'],
        'pidg_centroid_tuning':['G0_control','G1_centroid'],
        'pidg_v3_iso_scratch':['V3_scratch_control','V3_scratch_iso_1e3'],
    }.items():
        for name in names:
            directory=ROOT/'runs'/family/name/'rubber_duck'
            if not (directory/'results.json').exists():
                matches=list((ROOT/'runs').glob('*/'+name+'/rubber_duck/results.json'))
                if len(matches)!=1:raise ValueError(f'Cannot resolve historical model {name}: {matches}')
                directory=matches[0].parent
            context.append({'family':family,'name':name,'directory':str(directory),
                'test':read(directory/'results.json')['ours_50000'],'config':read(directory/'metric_config.json')})
    output={'protocol':{'scene':'Rubber Duck','split':[120,30,50],'seed':0,'iterations':50000,'snapshot':'final50000',
        'image_size':[1600,900],'frame_ids':frame_ids,'shared3k_sha256':shared[0],'all_GT_exports_identical':True,
        'test_exposure_historical':True,'time_task':'interleaved temporal interpolation, not held-out-block forecasting'},
        'models':models,'pairwise_vs_C1':pairwise,'object_id':object_id,'context_models':context,'sources':SOURCES}
    (ASSETS/'data.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={m['id']:{'test_PSNR':m['test']['PSNR'],'foreground_PSNR':m['photometric_mean']['foreground_PSNR'],
        'BG_mse':m['photometric_mean']['background_mse'],'score':m['integrity']['mean']['integrity_score'],'wins':pairwise[m['id']]['improved']} for m in models}
    print(json.dumps({'models':summary,'H3_delta':pairwise['H3'],'object_id_diagnostics':[{k:x['diagnostic'].get(k) for k in ('model','mean_max_assignment','confident_kernel_fraction')} for x in object_id]},ensure_ascii=False,indent=2))
    return output


if __name__=='__main__':collect()
