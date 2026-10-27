"""Paired scratch 50k centroid ablation, shared pre-activation full checkpoint."""
import argparse
from datetime import datetime, timedelta, timezone
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from variants.coverage_tuning import run as coverage
from variants.v3_iso_scratch import run as scratch
from variants.centroid_tuning.prepare import prepare
from variants.coverage_tuning.data import audit_training, read_alpha

HERE = Path(__file__).resolve().parent
RUNS = ROOT / 'runs/pidg_centroid_tuning'
REPORTS = ROOT / 'experiments/centroid_tuning'
NAMES = ['G0_control', 'G1_centroid']
BASE = 'original'


def configuration(name, smoke=False, off=False):
    c, text = scratch.configuration(scratch.NAMES[0], smoke, off)
    if BASE == 'original':
        c.update(perceptual_weight=0., shared_physics_time=False, timeline='official_metadata')
    c.update(baseline_kind=BASE, centroid_weight=1. if name==NAMES[1] and not off else 0.,
             centroid_start=20 if smoke else 3000, centroid_ramp=10 if smoke else 2000,
             centroid_eps=1e-6, mask_weight=0., affine_weight=0., coverage_density=False)
    return c, text


def freeze(repo):
    return {'parent': coverage.freeze(coverage.HERE/'code'), 'code': engine.runtime_hashes(repo),
            'inputs': {str(p.relative_to(ROOT)): engine.sha256(p) for p in [*HERE.glob('*.py'), HERE/'launch.ps1']},
            'alpha_audit_sha256': engine.sha256(REPORTS/'alpha_audit.json')}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json') != freeze(repo):
        raise ValueError('Centroid source/data changed')


def wire(repo):
    base.RUNS, base.REPORTS, base.assert_frozen = RUNS, REPORTS, assert_frozen
    engine.RUNS, engine.REPORTS, engine.repo_global, engine.assert_frozen = RUNS, REPORTS, repo, assert_frozen


def initialize(name, smoke=False, off=False):
    directory = RUNS / ('_smoke' if smoke else '') / name / 'rubber_duck'
    c, text = configuration(name, smoke, off)
    if (directory/'reproduction.json').exists():
        if engine.read_json(directory/'metric_config.json') != c or (directory/'model_config.py').read_text() != text:
            raise ValueError('Immutable centroid configuration changed')
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    engine.write_json(directory/'metric_config.json', c)
    engine.write_text_atomic(directory/'model_config.py', text)
    engine.write_json(directory/'reproduction.json', {'id': name, 'commands': [], 'status': 'initialized',
        'baseline_kind': BASE, 'warm_start': False, 'parent': None,
        'target_iterations':100 if smoke else 50000, 'resolution_width':400 if smoke else 1600,
        'train':120, 'validation':30, 'test':50})
    return directory


def step(directory):
    path = directory/'checkpoints/resume.json'
    return engine.read_json(path)['iteration'] if path.exists() else -1


def branch(source, destination, boundary):
    receipt = destination/'branch.json'
    if receipt.exists():
        old = engine.read_json(receipt)
        if old['source'] != str(source) or old['iteration'] != boundary:
            raise ValueError('Branch source changed')
        return
    if (destination/'checkpoints/resume.pt').exists():
        raise ValueError('Unreceipted checkpoint at destination')
    a, b = [engine.read_json(d/'metric_config.json') for d in (source, destination)]
    left, right = dict(a), dict(b)
    if left.pop('centroid_weight') != 0. or right.pop('centroid_weight') not in (0.,1.) or left != right:
        raise ValueError('Only centroid_weight may change at the common boundary')
    if step(source) != boundary:
        raise ValueError('Wrong shared branch iteration')
    for folder in ('point_cloud', 'deform', 'checkpoints'):
        shutil.copytree(source/folder, destination/folder, dirs_exist_ok=True)
    for filename in ('metric_best.json', 'metric_history.jsonl'):
        shutil.copyfile(source/filename, destination/filename)
    checkpoint = destination/'checkpoints/resume.pt'
    digest = engine.sha256(checkpoint)
    if digest != engine.read_json(source/'checkpoints/resume.json')['sha256']:
        raise ValueError('Full state copy changed')
    sidecar = checkpoint.with_suffix('.metrics.json'); state = engine.read_json(sidecar)
    if state['config'] != a:
        raise ValueError('Unexpected source sidecar config')
    state['config'] = b; engine.write_json(sidecar, state)
    engine.write_json(receipt, {'source': str(source), 'destination': str(destination), 'iteration': boundary,
        'identical_full_checkpoint_sha256': digest, 'source_config': a, 'branch_config': b,
        'only_change': 'centroid_weight 0 -> '+str(b['centroid_weight']),
        'initialization': 'shared random step0 followed by shared pre-activation training',
        'created_utc': engine.now()})


def shared_prefix(repo, deadline, smoke=False):
    common = initialize('shared_prefix', smoke)
    boundary = 20 if smoke else 3000
    if step(common) < 0:
        os.environ['PIDG_CAPTURE_INITIAL_ONLY'] = '1'
        try:
            if base.train(common, repo, deadline, smoke) != 75:
                raise ValueError('Expected full random initial capture')
        finally:
            os.environ.pop('PIDG_CAPTURE_INITIAL_ONLY', None)
        from reproduction.scripts.training_checkpoint import load_state
        scratch.validate_initial_state(load_state(common/'checkpoints/resume.pt'))
        initial = {'iteration':0,'optimizer_updates':0,'pretrained_model_used':False,
                   'sha256':engine.sha256(common/'checkpoints/resume.pt'),'baseline_kind':BASE}
        engine.write_json(REPORTS/('smoke_initialization.json' if smoke else 'initialization.json'), initial)
    if step(common) < boundary:
        if base.train(common, repo, deadline, smoke, limit=boundary) != 75:
            raise ValueError('Shared prefix did not stop at branch boundary')
    if step(common) != boundary:
        raise ValueError('Common prefix must remain frozen at branch iteration')
    arms = [initialize(name, smoke) for name in NAMES]
    for destination in arms:
        branch(common, destination, boundary)
    return arms


def centroid_scores(directory, split='test', tags=None):
    import torch
    import numpy as np
    from PIL import Image
    from variants.centroid_tuning.model import foreground_centroid
    source = base.DATA/('final_test' if split=='test' else 'training')
    frames = engine.read_json(source/'transforms_test.json')['frames']
    values = {}
    for tag in tags or base.TAGS:
        rows = []
        for i, frame in enumerate(frames):
            raw = directory/f'test/ours_{tag}/alpha/{i:05d}.png'
            with Image.open(raw) as image:
                alpha = torch.from_numpy(np.array(image.convert('L'), dtype=np.float32).copy())[None]/255.
                size = image.size
            target = read_alpha(source/(frame['file_path']+'.png'), size)
            delta = foreground_centroid(alpha)-foreground_centroid(target)
            rows.append({'frame':Path(frame['file_path']).stem, 'centroid_distance':delta.norm().item(),
                         'centroid_squared_error':delta.square().sum().item(),
                         'predicted_mean_alpha':alpha.mean().item()})
        values[tag] = {'count':len(rows),'per_frame':rows,
            'mean':{key:sum(r[key] for r in rows)/len(rows) for key in rows[0] if key!='frame'}}
    engine.write_json(directory/('centroid_metrics_'+split+'.json'), values)
    return values


def report():
    state = engine.read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {}
    active = engine.read_json(RUNS/'active.json') if (RUNS/'active.json').exists() else {}
    rows = []
    for name in ['shared_prefix', *NAMES]:
        d = RUNS/name/'rubber_duck'; history = base.history(d)
        rows.append({'id':name,'iteration':base.progress(d),'history':history,
            'test':engine.read_json(d/'results.json').get('ours_50000') if (d/'results.json').exists() else None,
            'coverage':engine.read_json(d/'coverage_metrics_test.json').get('50000',{}).get('mean') if (d/'coverage_metrics_test.json').exists() else None,
            'centroid':engine.read_json(d/'centroid_metrics_test.json').get('50000',{}).get('mean') if (d/'centroid_metrics_test.json').exists() else None})
    comparison = None
    if all(row['test'] for row in rows[1:]):
        comparison = {key:rows[2]['test'][key]-rows[1]['test'][key] for key in engine.METRICS}
    result = {'updated_utc':engine.now(),'baseline_kind':BASE,'controller':state,'active':active,
              'rows':rows,'G1_minus_G0_final50k':comparison,'split':[120,30,50],
              'scope':'whole-scene training-alpha centroid, not physical 3D mass or object-separated supervision',
              'limitations':['historical test exposure','single scene and seed','CUDA nondeterminism after branch',
                             'global centroid can hide compensating object errors and internal holes']}
    engine.write_json(REPORTS/'results.json', result)
    lines = ['# Original PIDG + foreground centroid ablation','',
        'Base: '+BASE+' | Status: '+state.get('status','preparing'),
        'Active: '+active.get('id','-')+' / '+active.get('phase','-'),'',
        '120 train /30 validation /50 fixed test; two50k total arms from one shared random start and full3k state.',
        'G0: center weight0. G1: center weight1, start after3k, ramp2k. No mask L1, affine, iso, kinematic or late density additions.',
        'Original PIDG objective/physics-time convention on the clean temporal split; V3 is a separately named optional base.',
        'This is a 2D whole-scene alpha centroid, not a measured physical3D center of mass. Training targets never use validation/test alpha.', '',
        '| Stage | Current iter | Latest validation PSNR | Test PSNR |', '|---|---:|---:|---:|']
    for row in rows:
        val = row['history'][-1]['metrics']['PSNR'] if row['history'] else '-'
        lines.append(f"| {row['id']} | {row['iteration']}/{'3000' if row['id']=='shared_prefix' else '50000'} | {val} | {(row['test'] or {}).get('PSNR','-')} |")
    if comparison:
        lines += ['', 'Final50k G1 minus G0: '+json.dumps(comparison)]
    lines += ['', '[5-second graphical monitor](monitor.html) | [Plan](plan.json) | [Preflight](preflight.json)',
              '', 'GT/render galleries: '+str(RUNS/'render_comparisons'), '', 'Updated UTC: '+result['updated_utc']]
    engine.write_text_atomic(REPORTS/'results.md', '\n'.join(lines)+'\n')
    page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>PIDG centroid ablation</title><style>body{font:16px system-ui;background:#111c2c;color:#edf5ff;margin:30px}section,svg{background:#203047;padding:12px;margin:12px 0}progress{width:85%}svg{max-width:100%}a{color:#89dfba}</style><h1>PIDG + training foreground centroid</h1><p>'+html.escape(BASE+' | '+state.get('status','preparing')+' | '+active.get('id','-')+' / '+active.get('phase','-'))+'</p><p>50k total each; shared random initialization and3k prefix. Refresh5s. Whole-scene2D centroid; no physical mass labels.</p>'
    for row in rows:
        target = 3000 if row['id']=='shared_prefix' else 50000
        page += f"<section><b>{row['id']}</b><h2>{row['iteration']:,} / {target:,} iter</h2><progress value='{row['iteration']}' max='{target}'></progress></section>"
    for metric in engine.METRICS:
        values = [h['metrics'][metric] for row in rows[1:] for h in row['history']]
        if not values:
            continue
        lo,hi=min(values),max(values); span=max(hi-lo,1e-6)
        page += f'<h2>Validation {metric}</h2><svg viewBox="0 0 900 270" width="900"><text x="5" y="20" fill="white">{hi:.5f}</text><text x="5" y="220" fill="white">{lo:.5f}</text>'
        for i,(row,color) in enumerate(zip(rows[1:],['#8fd3ff','#85e5b5'])):
            points = ' '.join(f"{80+790*h['iteration']/50000:.1f},{220-190*(h['metrics'][metric]-lo)/span:.1f}" for h in row['history'])
            page += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="{80+i*400}" y="255" fill="{color}">{row["id"]}</text>'
        page += '</svg>'
    engine.write_text_atomic(REPORTS/'monitor.html', page+'<p><a href="results.md">Results and protocol</a></p>')


def preflight(repo, deadline):
    environment = os.environ.copy(); environment['PIDG_CENTROID_GPU_TEST']='1'
    result = subprocess.run([sys.executable,'-m','pytest','tests/test_centroid_rasterizer_gpu.py','-q'],
                            cwd=ROOT,env=environment,capture_output=True,text=True)
    engine.write_text_atomic(REPORTS/'gpu_gradient_test.log',result.stdout+result.stderr)
    if result.returncode:
        raise ValueError('Real renderer centroid gradient test failed')
    off = initialize('flags_off',True,True)
    if step(off)<20:
        if base.train(off,repo,deadline,True,off=True)!=0:
            raise ValueError('Flags-off smoke failed')
    arms = shared_prefix(repo,deadline,True)
    for d in arms:
        if step(d)<100 and base.train(d,repo,deadline,True)!=0:
            raise ValueError('Paired smoke training failed')
    if (arms[0]/'centroid_loss.jsonl').exists():
        raise ValueError('Control must not compute centroid loss')
    gradients=[json.loads(s) for s in (arms[1]/'centroid_gradient.jsonl').read_text().splitlines()]
    if not any(r['alpha_l1']>0 for r in gradients):
        raise ValueError('Centroid gradient missing in real training')
    if 'FULL_CHECKPOINT_RESTORED completed=20 next=21' not in (arms[1]/'train.log').read_text():
        raise ValueError('Shared full-state resume not exercised')
    for d in arms:
        base.render(d,repo,deadline,split='validation',tags=['best'],smoke=True)
        centroid_scores(d,'validation',['best'])
    engine.write_json(REPORTS/'preflight.json',{'status':'passed','flags_off_steps':20,
        'smoke_steps_per_arm':100,'shared_branch_iteration':20,'train_only_targets':True,
        'actual_cuda_gradient_test':True,'finite_centroid_gradients':gradients,
        'validation_png_parity':True,'source_freeze_sha256':engine.sha256(REPORTS/'source_freeze.json'),
        'completed_utc':engine.now()})


def main():
    global BASE
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--hours',type=float,default=24.); parser.add_argument('--base',choices=['original','v3'],default='original')
    args=parser.parse_args(); BASE=args.base
    if args.hours<=0:
        raise ValueError('Positive time budget required')
    RUNS.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True)
    coverage.assert_frozen(coverage.HERE/'code'); repo=prepare()
    if not (REPORTS/'alpha_audit.json').exists():
        engine.write_json(REPORTS/'alpha_audit.json',audit_training(base.DATA/'training'))
    if (REPORTS/'source_freeze.json').exists(): assert_frozen(repo)
    else: engine.write_json(REPORTS/'source_freeze.json',freeze(repo))
    wire(repo)
    for name in NAMES:
        c,text=configuration(name)
        engine.write_json(REPORTS/'configs'/(name+'.json'),c)
        engine.write_text_atomic(REPORTS/'configs'/(name+'.py'),text)
    plan={'baseline_kind':BASE,'arms':{n:configuration(n)[0] for n in NAMES},'seed':0,
        'initialization':'random step0; same complete3k state before centroid activation; no pretrained parent',
        'iterations_total_each':50000,'shared_prefix_iterations':3000,'split':[120,30,50],
        'loss':'squared L2 between normalized XY alpha centroids, weight1, start after3000, ramp2000',
        'targets':'only original120 training RGBA alphas; whole-scene union, no per-object labels',
        'selection':'validation only; lock both arms before final test export','test_tags':base.TAGS,
        'reuse_note':'Historical final-only controls lack the matched full3k state; new matched control is required.',
        'physical_mass_claim':False,'historical_test_exposure':True,'budget_hours':args.hours}
    if (REPORTS/'plan.json').exists() and engine.read_json(REPORTS/'plan.json')!=plan:
        raise ValueError('Immutable experiment plan changed')
    engine.write_json(REPORTS/'plan.json',plan)
    deadline=datetime.now(timezone.utc)+timedelta(hours=args.hours)
    state={'pid':os.getpid(),'status':'preflight','started_utc':engine.now(),
           'deadline_utc':deadline.isoformat(),'mode':'smoke' if args.smoke else 'long','completed':[]}
    stop=threading.Event()
    def watch():
        while not stop.wait(5):
            try: report()
            except (OSError,ValueError,KeyError): pass
    with engine.lock():
        if (RUNS/'stop_queue.request').exists():
            raise ValueError('User pause request remains')
        engine.write_json(RUNS/'controller.json',state); report()
        thread=threading.Thread(target=watch,daemon=True); thread.start()
        os.environ['PIDG_EXPORT_ALPHA']='1'
        try:
            proof=engine.read_json(REPORTS/'preflight.json') if (REPORTS/'preflight.json').exists() else {}
            if proof.get('source_freeze_sha256')!=engine.sha256(REPORTS/'source_freeze.json'):
                preflight(repo,deadline)
            if args.smoke:
                state['status']='preflight_passed'
            else:
                state['status']='shared_prefix'; engine.write_json(RUNS/'controller.json',state)
                arms=shared_prefix(repo,deadline)
                for d in arms:
                    state['status']='training'; engine.write_json(RUNS/'controller.json',state)
                    if step(d)<50000 and base.train(d,repo,deadline)==75:
                        raise SystemExit(75)
                    state['completed'].append(d.parent.name); engine.write_json(RUNS/'controller.json',state)
                selected={d.parent.name:engine.read_json(d/'metric_best.json') for d in arms}
                lock=RUNS/'selection_lock.json'
                if lock.exists() and engine.read_json(lock)['validation_selections']!=selected:
                    raise ValueError('Locked selections changed')
                engine.write_json(lock,{'locked_utc':engine.now(),'validation_selections':selected,'tags':base.TAGS})
                state['status']='final_test'; engine.write_json(RUNS/'controller.json',state)
                for d in arms:
                    base.render(d,repo,deadline); base.pairs(d)
                    coverage.coverage_scores(d); centroid_scores(d)
                state['status']='completed'
            assert_frozen(repo)
        except SystemExit as error:
            state['status']='saved_stopped' if error.code==75 else 'failed'
            if error.code!=75: raise
        except Exception as error:
            state.update(status='failed',error=str(error)); raise
        finally:
            state['finished_utc']=engine.now(); engine.write_json(RUNS/'controller.json',state)
            stop.set(); thread.join(timeout=6); report()


if __name__=='__main__': main()
