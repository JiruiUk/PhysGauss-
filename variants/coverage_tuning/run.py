"""Ordered 50k scratch ablations; validation selection before locked final test."""
import argparse
from datetime import datetime, timedelta, timezone
import html
import json
import os
from pathlib import Path
import shutil
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from variants.v3_iso_scratch import run as scratch
from variants.coverage_tuning.prepare import prepare
from variants.coverage_tuning.data import audit_training

HERE = Path(__file__).resolve().parent
RUNS = ROOT/'runs/pidg_coverage_tuning'
REPORTS = ROOT/'experiments/coverage_tuning'
NAMES = ['C0_V3_reference','C1_mask','C2_mask_affine','C3_mask_affine_density']
TRAIN_NAMES = NAMES[1:]
TAGS = base.TAGS


def configuration(name,smoke=False,off=False):
    config,text = scratch.configuration(scratch.NAMES[0],smoke,off)
    index = NAMES.index(name) if name in NAMES else 0
    config.update(mask_weight=.1 if index >= 1 else 0.,affine_weight=1e-4 if index >= 2 else 0.,
        coverage_density=index >= 3,single_object=True,affine_start=0 if smoke else 3000,
        affine_ramp=1 if smoke else 2000,affine_interval=10,affine_samples=32 if smoke else 128,
        affine_radius=.1,coverage_start=20 if smoke else 16000,coverage_until=100 if smoke else 36000,
        coverage_sample_interval=1 if smoke else 20,coverage_window=20 if smoke else 2000,
        coverage_min_views=1 if smoke else 3,coverage_add_limit=8 if smoke else 512,
        coverage_max_points=200000)
    return config,text


def freeze(repo):
    paths = list(HERE.glob('*.py'))+[HERE/'launch.ps1']
    return {'scratch':scratch.freeze(scratch.HERE/'code'),'code':engine.runtime_hashes(repo),
        'inputs':{str(p.relative_to(ROOT)):engine.sha256(p) for p in paths},
        'training_alpha':engine.read_json(REPORTS/'alpha_audit.json'),
        'rgba_file_stat':{r['path']:[Path(r['path']).stat().st_size,Path(r['path']).stat().st_mtime_ns]
                          for r in engine.read_json(REPORTS/'alpha_audit.json')['frames']}}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json') != freeze(repo):
        raise ValueError('Coverage experiment source/data changed')


def wire(repo):
    base.RUNS,base.REPORTS,base.assert_frozen = RUNS,REPORTS,assert_frozen
    engine.RUNS,engine.REPORTS,engine.repo_global,engine.assert_frozen = RUNS,REPORTS,repo,assert_frozen


def initialize(name,smoke=False,off=False):
    directory = RUNS/('_smoke' if smoke else '')/name/'rubber_duck'
    config,text = configuration(name,smoke,off)
    if (directory/'reproduction.json').exists():
        if engine.read_json(directory/'metric_config.json') != config or (directory/'model_config.py').read_text() != text:
            raise ValueError('Immutable coverage config changed')
        return directory
    directory.mkdir(parents=True,exist_ok=True)
    engine.write_json(directory/'metric_config.json',config)
    engine.write_text_atomic(directory/'model_config.py',text)
    engine.write_json(directory/'reproduction.json',{'id':name,'warm_start':False,'parent':None,
        'target_iterations':100 if smoke else 50000,'resolution_width':400 if smoke else 1600,
        'train':120,'validation':30,'test':50,'commands':[],'status':'initialized'})
    return directory


def clone_zero(destination,smoke=False):
    source = ROOT/'runs/pidg_v3_iso_scratch'/('_smoke' if smoke else '')/'initialization/rubber_duck'
    receipt = destination/'branch.json'
    if receipt.exists(): return
    from reproduction.scripts.training_checkpoint import load_state
    scratch.validate_initial_state(load_state(source/'checkpoints/resume.pt'))
    # Enforce the exact same original scratch optimizer recipe; only this
    # experiment's opt-in objective/density flags may differ in the sidecar.
    if (source/'model_config.py').read_text() != (destination/'model_config.py').read_text():
        raise ValueError('Step-zero optimizer/network recipe changed')
    for folder in ('point_cloud','deform','checkpoints'):
        shutil.copytree(source/folder,destination/folder,dirs_exist_ok=True)
    for file in ('metric_best.json','metric_history.jsonl'):
        shutil.copyfile(source/file,destination/file)
    checkpoint = destination/'checkpoints/resume.pt'
    digest = engine.sha256(checkpoint)
    if digest != engine.read_json(source/'checkpoints/resume.json')['sha256']: raise ValueError('Zero state copy changed')
    path = checkpoint.with_suffix('.metrics.json'); state = engine.read_json(path)
    state['config'] = engine.read_json(destination/'metric_config.json'); engine.write_json(path,state)
    engine.write_json(receipt,{'source':str(source),'destination':str(destination),'iteration':0,
        'identical_full_checkpoint_sha256':digest,'optimizer_updates':0,'pretrained_model_used':False,
        'config':state['config'],'created_utc':engine.now()})


def reference():
    """Copy inference snapshots, never mutate/retrain the completed control."""
    directory = initialize(NAMES[0])
    source = ROOT/'runs/pidg_v3_iso_scratch/V3_scratch_control/rubber_duck'
    receipt = directory/'reference.json'
    if receipt.exists(): return directory
    if engine.read_json(source/'checkpoints/resume.json')['iteration'] != 50000:
        raise ValueError('Reference 50k training is incomplete')
    for folder in ('point_cloud','deform'):
        for tag in TAGS:
            shutil.copytree(source/folder/('iteration_'+tag),directory/folder/('iteration_'+tag),dirs_exist_ok=True)
    for name in ('metric_best.json','metric_history.jsonl','cfg_args'):
        shutil.copyfile(source/name,directory/name)
    engine.write_json(receipt,{'source':str(source),'iteration':50000,'reused':True,
        'initialization':engine.read_json(ROOT/'experiments/v3_iso_scratch/initialization.json'),
        'checkpoint_sha256':engine.sha256(source/'checkpoints/resume.pt'),
        'original_metrics':engine.read_json(source/'results.json'),'created_utc':engine.now()})
    return directory


def coverage_scores(directory,split='test',tags=None):
    from PIL import Image
    import numpy as np
    import torch
    from variants.coverage_tuning.data import read_alpha
    from variants.coverage_tuning.model import coverage_metrics
    source = base.DATA/('final_test' if split == 'test' else 'training')
    frames = engine.read_json(source/'transforms_test.json')['frames']
    result = {}
    for tag in tags or TAGS:
        raw = directory/'test'/('ours_'+tag)
        rows = []
        for i,frame in enumerate(frames):
            name = f'{i:05d}.png'
            def rgb(folder):
                with Image.open(raw/folder/name) as im:
                    return torch.from_numpy(np.array(im.convert('RGB'),dtype=np.float32).copy()).permute(2,0,1)/255.
            with Image.open(raw/'alpha'/name) as im:
                a = torch.from_numpy(np.array(im.convert('L'),dtype=np.float32).copy())[None]/255.
                size = im.size
            target = read_alpha(source/(frame['file_path']+'.png'),size)
            rows.append({'frame':Path(frame['file_path']).stem,**coverage_metrics(a,target,rgb('renders'),rgb('gt'))})
        result[tag] = {'split':split,'count':len(rows),'per_frame':rows,
            'mean':{k:sum(r[k] for r in rows)/len(rows) for k in rows[0] if k != 'frame'}}
    engine.write_json(directory/('coverage_metrics_'+split+'.json'),result)
    return result


def report():
    state = engine.read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {}
    active = engine.read_json(RUNS/'active.json') if (RUNS/'active.json').exists() else {}
    rows = []
    for name in NAMES:
        p = RUNS/name/'rubber_duck'; history = base.history(p)
        rows.append({'id':name,'iteration':50000 if (p/'reference.json').exists() else base.progress(p),
            'history':history,'test':engine.read_json(p/'results.json') if (p/'results.json').exists() else None,
            'coverage':engine.read_json(p/'coverage_metrics_test.json') if (p/'coverage_metrics_test.json').exists() else None})
    result = {'updated_utc':engine.now(),'controller':state,'active':active,'rows':rows,
        'protocol':'Ordered additions; same random zero state; 50k total; iso=0; train120/val30/test50',
        'mask_supervision':'only 120 training RGBA alphas','test_selection':'validation only; all arms locked before test',
        'limitations':['single scene/seed','historical official test exposure','CUDA nondeterminism',
            'no-contributor holes cannot seed 3D geometry','single-object neighborhoods only',
            'mask/coverage is not a guarantee of watertight or correct 3D geometry']}
    engine.write_json(REPORTS/'results.json',result)
    lines = ['# Ordered foreground coverage ablation','',result['protocol'],'',
        'Status: '+state.get('status','preparing'),'Active: '+active.get('id','-')+' / '+active.get('phase','-'),'',
        '| Model | Iteration | Mask weight | Local affine weight | Coverage densify | Latest validation PSNR |',
        '|---|---:|---:|---:|---|---:|']
    for row in rows:
        c,_ = configuration(row['id'])
        lines.append(f"| {row['id']} | {row['iteration']}/50000 | {c['mask_weight']} | {c['affine_weight']} | {c['coverage_density']} | {row['history'][-1]['metrics']['PSNR'] if row['history'] else '-'} |")
    lines += ['','## Locked final50k test comparison','',
        '| Model | PSNR | SSIM | LPIPS | MS-SSIM | Foreground PSNR | Mask IoU | Interior hole rate |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['test']:
            m = row['test']['ours_50000']; extra = row['coverage']['50000']['mean'] if row['coverage'] else {}
            lines.append('| '+row['id']+' | '+' | '.join(f'{m[k]:.7f}' for k in engine.METRICS)+' | '+
                ' | '.join(str(extra.get(k,'pending')) for k in ('foreground_psnr','mask_iou','interior_hole_rate'))+' |')
    lines += ['','Training masks: original RGBA alpha, L1 mean over all pixels, weight0.1 from step1.',
        'Local affine: weight1e-4, every10 steps with interval compensation, starts3000/ramp2000; permits affine stretch/shear.',
        'Coverage seeds: contributors to eroded foreground alpha<0.7, target0.95, sampled every20 steps after16000; >=3 distinct training times.',
        'At18000..36000 every2000: add <=512 points, cap200000; keep parents; no extra pruning/reset. Original densification ends15000.',
        'Coverage-only metrics use threshold0.5 and 3-pixel eroded GT interior; foreground PSNR uses alpha>0.5.',
        'The reference reuses completed V3 scratch control; all candidates start from identical random iteration0, not from the reference weights.',
        'Final test remains historically exposed. Single seed; paired arm comparisons do not establish statistical significance.',
        '', 'Render / GT pairs absolute directory: '+str(RUNS/'render_comparisons'),
        '', '[5-second charts](monitor.html) | [Plan](plan.json) | [Training alpha audit](alpha_audit.json)',
        '', 'Updated UTC: '+result['updated_utc']]
    engine.write_text_atomic(REPORTS/'results.md','\n'.join(lines)+'\n')
    page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>Coverage ablation</title><style>body{font:16px system-ui;background:#122033;color:#edf5ff;margin:30px}section,svg{background:#203149;padding:12px;margin:12px 0}progress{width:85%}a{color:#86e1bb}</style><h1>Coverage ablation / 5-second refresh</h1><p>'+html.escape(state.get('status','preparing')+' | '+active.get('id','-')+' | '+active.get('phase','-'))+'</p>'
    colors = ['#a8b1bd','#82cfff','#94e9b2','#ffbe83']
    for row in rows:
        page += f"<section><b>{row['id']}</b><h2>{row['iteration']:,} / 50,000 iter</h2><progress value='{row['iteration']}' max='50000'></progress></section>"
    for key in engine.METRICS:
        values = [h['metrics'][key] for r in rows for h in r['history']]
        if not values: continue
        lo,hi = min(values),max(values); span = max(hi-lo,1e-8)
        page += f'<h2>Validation {key}</h2><svg viewBox="0 0 900 300" width="900"><text x="5" y="20" fill="white">{hi:.5f}</text><text x="5" y="220" fill="white">{lo:.5f}</text>'
        for i,(r,color) in enumerate(zip(rows,colors)):
            points = ' '.join(f"{80+790*h['iteration']/50000:.1f},{220-190*(h['metrics'][key]-lo)/span:.1f}" for h in r['history'])
            page += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="{80+(i%2)*400}" y="{250+(i//2)*22}" fill="{color}">{r["id"]}</text>'
        page += '</svg>'
    page += '<p><a href="results.md">Results / definitions / absolute render paths</a></p>'
    engine.write_text_atomic(REPORTS/'monitor.html',page)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--hours',type=float,default=24)
    args = parser.parse_args()
    if not math_is_positive(args.hours): raise ValueError('Positive finite hours required')
    RUNS.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True)
    scratch.assert_frozen(scratch.HERE/'code')
    repo = prepare()
    if not (REPORTS/'alpha_audit.json').exists(): engine.write_json(REPORTS/'alpha_audit.json',audit_training(base.DATA/'training'))
    if (REPORTS/'source_freeze.json').exists(): assert_frozen(repo)
    else: engine.write_json(REPORTS/'source_freeze.json',freeze(repo))
    wire(repo)
    os.environ['PIDG_EXPORT_ALPHA'] = '1'
    engine.write_json(REPORTS/'plan.json',{'order':NAMES,'reference_reused':True,'total_steps':50000,
        'initialization':'same scratch random iteration0; no trained parent','iso_weight':0,
        'split':[120,30,50],'supervised_alpha_split':'train only','test_tags':TAGS,'seed':0,
        'configs':{n:configuration(n)[0] for n in NAMES},'hours':args.hours,
        'selection':'validation-only 4 best + fixed50k; lock all before test',
        'extra_metrics':['foreground PSNR','mask IoU','interior low-alpha hole rate'],
        'render_pairs_absolute':str(RUNS/'render_comparisons')})
    for name in NAMES:
        c,t = configuration(name); engine.write_json(REPORTS/'configs'/(name+'.json'),c)
        engine.write_text_atomic(REPORTS/'configs'/(name+'.py'),t)
    deadline = datetime.now(timezone.utc)+timedelta(hours=args.hours)
    state = {'pid':os.getpid(),'status':'preflight' if args.smoke else 'training',
        'started_utc':engine.now(),'deadline_utc':deadline.isoformat(),'completed':[]}
    stop = threading.Event()
    def watch():
        while not stop.wait(5):
            try: report()
            except (OSError,ValueError,KeyError): pass
    def step(d):
        return engine.read_json(d/'checkpoints/resume.json')['iteration'] if (d/'checkpoints/resume.json').exists() else -1
    with engine.lock():
        if (RUNS/'stop_queue.request').exists(): raise RuntimeError('Pause request remains')
        engine.write_json(RUNS/'controller.json',state); report()
        watcher = threading.Thread(target=watch,daemon=True); watcher.start()
        try:
            if args.smoke:
                off = initialize('flags_off',True,True)
                if step(off)<20: assert base.train(off,repo,deadline,True,off=True)==0
            elif engine.read_json(REPORTS/'preflight.json')['status'] != 'passed': raise ValueError('GPU preflight required')
            directories = []
            for name in TRAIN_NAMES:
                d = initialize(name,args.smoke); clone_zero(d,args.smoke); directories.append(d)
                if args.smoke and name == NAMES[3] and step(d)<40:
                    assert base.train(d,repo,deadline,True,limit=40)==75
                if step(d)<(100 if args.smoke else 50000):
                    if base.train(d,repo,deadline,args.smoke)==75: raise SystemExit(75)
                state['completed'].append(name); engine.write_json(RUNS/'controller.json',state)
            if args.smoke:
                density = directories[-1]
                records = [json.loads(s) for s in (density/'coverage_density.jsonl').read_text().splitlines()]
                if not records or not any(r['added']>0 for r in records): raise ValueError('Smoke did not exercise real point additions')
                if 'FULL_CHECKPOINT_RESTORED completed=40 next=41' not in (density/'train.log').read_text(): raise ValueError('Density resume not exercised')
                for d in directories:
                    if not (d/'coverage_gradient.jsonl').exists(): raise ValueError('Missing alpha gradient')
                    if d != directories[0] and not (d/'affine_loss.jsonl').exists(): raise ValueError('Affine branch not exercised')
                base.render(directories[0],repo,deadline,split='validation',tags=['best'],smoke=True)
                coverage_scores(directories[0],'validation',['best'])
                engine.write_json(REPORTS/'preflight.json',{'status':'passed','flags_off_steps':20,
                    'all_three_arms_steps':100,'density_full_resume_from':40,'density_events':records,
                    'validation_png_parity':True,'alpha_gradients':True,'affine_branch_exercised':True,
                    'completed_utc':engine.now(),'source_freeze_sha256':engine.sha256(REPORTS/'source_freeze.json')})
                state['status'] = 'preflight_passed'
            else:
                control = reference(); directories.insert(0,control)
                selections = {d.parent.name:engine.read_json(d/'metric_best.json') for d in directories}
                lock = RUNS/'selection_lock.json'
                if lock.exists():
                    if engine.read_json(lock)['validation_selections'] != selections: raise ValueError('Locked selections changed')
                else: engine.write_json(lock,{'locked_utc':engine.now(),'validation_selections':selections,'tags':TAGS})
                state['status'] = 'final_test'; engine.write_json(RUNS/'controller.json',state)
                for d in directories:
                    base.render(d,repo,deadline); base.pairs(d); coverage_scores(d)
                expected = engine.read_json(control/'reference.json')['original_metrics']['ours_50000']
                actual = engine.read_json(control/'results.json')['ours_50000']
                if any(abs(expected[k]-actual[k])>1e-4 for k in engine.METRICS): raise ValueError('Reference RGB parity failure')
                state['status'] = 'completed'
            assert_frozen(repo)
        except SystemExit as error:
            state['status'] = 'saved_stopped' if error.code == 75 else 'failed'
            if error.code != 75: raise
        except Exception as error:
            state.update(status='failed',error=str(error)); raise
        finally:
            state['finished_utc'] = engine.now(); engine.write_json(RUNS/'controller.json',state)
            stop.set(); watcher.join(timeout=6); report()


def math_is_positive(value):
    import math
    return math.isfinite(value) and value>0


if __name__ == '__main__': main()
