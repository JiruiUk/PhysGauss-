"""Queue two independent50k component ablations after the current GPU study."""
import argparse
from datetime import datetime,timezone
import html
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from variants.coverage_tuning import run as old
from variants.kinematic_resplit import run as base
from variants.metric_tuning import run as engine
from variants.coverage_isolated.prepare import prepare

HERE = Path(__file__).resolve().parent
RUNS = ROOT/'runs/pidg_coverage_isolated'
REPORTS = ROOT/'experiments/coverage_isolated'
NAMES = ['C4_affine_only','C5_density_only']


def configuration(name,smoke=False,off=False):
    c,text = old.configuration('C3_mask_affine_density',smoke,off)
    c.update(mask_weight=0.,affine_weight=1e-4 if name==NAMES[0] and not off else 0.,
        coverage_density=name==NAMES[1] and not off,ack_unsegmented_affine=name==NAMES[0] and not off)
    c.pop('single_object')
    return c,text


def freeze(repo):
    return {'coverage_parent':old.freeze(old.HERE/'code'),'code':engine.runtime_hashes(repo),
        'inputs':{str(p.relative_to(ROOT)):engine.sha256(p) for p in [*HERE.glob('*.py'),HERE/'launch.ps1']}}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json')!=freeze(repo): raise ValueError('Isolated source/data changed')


def wire(repo):
    base.RUNS,base.REPORTS,base.assert_frozen = RUNS,REPORTS,assert_frozen
    engine.RUNS,engine.REPORTS,engine.repo_global,engine.assert_frozen = RUNS,REPORTS,repo,assert_frozen


def initialize(name,smoke=False,off=False):
    d = RUNS/('_smoke' if smoke else '')/name/'rubber_duck'
    c,text = configuration(name,smoke,off)
    if (d/'reproduction.json').exists():
        if engine.read_json(d/'metric_config.json')!=c or (d/'model_config.py').read_text()!=text:
            raise ValueError('Existing isolated configuration changed')
        return d
    d.mkdir(parents=True,exist_ok=True)
    engine.write_json(d/'metric_config.json',c); engine.write_text_atomic(d/'model_config.py',text)
    engine.write_json(d/'reproduction.json',{'id':name,'warm_start':False,'parent':None,'commands':[],
        'target_iterations':100 if smoke else 50000,'resolution_width':400 if smoke else 1600,
        'train':120,'validation':30,'test':50,'status':'initialized'})
    return d


def step(d):
    p = d/'checkpoints/resume.json'
    return engine.read_json(p)['iteration'] if p.exists() else -1


def report():
    state = engine.read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {}
    active = engine.read_json(RUNS/'active.json') if (RUNS/'active.json').exists() else {}
    dependency = engine.read_json(old.RUNS/'controller.json')
    rows = []
    for name,root,components in [(n,old.RUNS,s) for n,s in zip(old.NAMES,
        ['V3 reference','mask','mask + affine','mask + affine + density'])]+[(NAMES[0],RUNS,'affine ONLY'),(NAMES[1],RUNS,'density ONLY')]:
        d = root/name/'rubber_duck'; h = base.history(d)
        rows.append({'id':name,'components':components,'iteration':50000 if (d/'reference.json').exists() else base.progress(d),
            'history':h,'test':engine.read_json(d/'results.json').get('ours_50000') if (d/'results.json').exists() else None,
            'coverage':engine.read_json(d/'coverage_metrics_test.json').get('50000',{}).get('mean') if (d/'coverage_metrics_test.json').exists() else None})
    comparisons = []
    if rows[0]['test']:
        for r in rows[1:]:
            if r['test']:
                comparisons.append({'id':r['id'],'reference':rows[0]['id'],
                    'final50k_delta':{k:r['test'][k]-rows[0]['test'][k] for k in engine.METRICS}})
    result = {'updated_utc':engine.now(),'controller':state,'active':active,'dependency':dependency,
        'rows':rows,'versus_V3':comparisons,'split':[120,30,50],'new_total_steps_per_arm':50000,
        'scope':'unsegmented affine-only vs train-alpha-guided density-only; zero mask loss in both',
        'limitations':['multi-object scene; no real object-ID isolation','single seed/scene',
            'historical and user-authorized early test exposure','CUDA nondeterminism','uniform frame cadence assumed']}
    engine.write_json(REPORTS/'results.json',result)
    lines = ['# Independent component effects: C4 / C5','',result['scope'],'',
        'Status: '+state.get('status','preparing'),'Dependency: '+dependency.get('status','-'),
        'Active: '+active.get('id','-')+' / '+active.get('phase','-'),'',
        'Both new arms start from the same random iteration0 as C0-C3;50k total; no pretrained parent.',
        'C4 loads no separate alpha target; baseline RGBA-to-RGB compositing is unchanged. C5 uses training alpha for point-addition selection, with L_mask=0.',
        'Affine formulation and density thresholds/schedule match the cumulative runs. No object-ID isolation is claimed.','',
        '| Model | Components | Iter | Validation PSNR | Test PSNR | SSIM | LPIPS | MS-SSIM | Foreground PSNR | Mask IoU | Interior hole rate |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        m,coverage = r['test'] or {},r['coverage'] or {}
        cells = [r['id'],r['components'],str(r['iteration'])+'/50000',
            str(round(r['history'][-1]['metrics']['PSNR'],5)) if r['history'] else '-']
        cells += [f'{m[k]:.7f}' if k in m else '-' for k in engine.METRICS]
        cells += [f'{coverage[k]:.7f}' if k in coverage else '-' for k in ('foreground_psnr','mask_iou','interior_hole_rate')]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['','Interpret independent effects as C4-C0 and C5-C0; mask-only remains C1-C0.',
        'C2-C1 and C3-C2 instead describe conditional effects when earlier components are enabled.',
        'Checkpoint choices use validation only; lock C4/C5 selections before their final test exports.',
        'Prior and early test exposure remains a limitation; new runs do not create a fresh untouched test set.',
        '', 'GT/render pairs absolute directory: '+str(RUNS/'render_comparisons'),
        '', '[5-second monitor](monitor.html) | [Plan](plan.json) | [GPU preflight](preflight.json)',
        '', 'Updated UTC: '+result['updated_utc']]
    engine.write_text_atomic(REPORTS/'results.md','\n'.join(lines)+'\n')
    page = '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>Independent components</title><style>body{font:16px system-ui;background:#122033;color:#edf5ff;margin:30px}section,svg{background:#203149;padding:12px;margin:12px 0}progress{width:85%}a{color:#86e1bb}</style><h1>C4 affine ONLY / C5 density ONLY</h1><p>120 train /30 validation /50 test;50k scratch each; refresh5s.</p><p>'+html.escape(state.get('status','preparing')+' | '+active.get('id','-')+' | '+active.get('phase','-'))+'</p><p>Unsegmented multi-object neighborhoods. Density-only uses training alpha for additions, no mask loss.</p>'
    for r in rows:
        page += f"<section><b>{r['id']} | {r['components']}</b><h2>{r['iteration']:,} /50,000 iter</h2><progress value='{r['iteration']}' max='50000'></progress></section>"
    colors = ['#a8b1bd','#82cfff','#94e9b2','#ffbe83','#e0a5ff','#ff8198']
    for key in engine.METRICS:
        values = [h['metrics'][key] for r in rows for h in r['history']]
        if not values: continue
        lo,hi = min(values),max(values); span = max(hi-lo,1e-8)
        page += f'<h2>Validation {key}</h2><svg viewBox="0 0 900 340" width="900"><text x="5" y="20" fill="white">{hi:.5f}</text><text x="5" y="220" fill="white">{lo:.5f}</text>'
        for i,(r,color) in enumerate(zip(rows,colors)):
            points = ' '.join(f"{80+790*h['iteration']/50000:.1f},{220-190*(h['metrics'][key]-lo)/span:.1f}" for h in r['history'])
            page += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="{80+(i%2)*400}" y="{250+(i//2)*25}" fill="{color}">{r["id"]}</text>'
        page += '</svg>'
    page += '<p><a href="results.md">All six variants / independent comparisons / metrics</a></p>'
    engine.write_text_atomic(REPORTS/'monitor.html',page)


def preflight(repo,deadline):
    off = initialize('flags_off',True,True)
    if step(off)<20: assert base.train(off,repo,deadline,True,off=True)==0
    dirs = []
    for name in NAMES:
        d = initialize(name,True); old.clone_zero(d,True); dirs.append(d)
        if name==NAMES[1] and step(d)<40: assert base.train(d,repo,deadline,True,limit=40)==75
        if step(d)<100: assert base.train(d,repo,deadline,True)==0
    affine = [json.loads(s) for s in (dirs[0]/'affine_gradient.jsonl').read_text().splitlines()]
    events = [json.loads(s) for s in (dirs[1]/'coverage_density.jsonl').read_text().splitlines()]
    if not any(r['trajectory_l1']>0 for r in affine): raise ValueError('Missing affine-only gradient')
    if not any(r['added']>0 for r in events): raise ValueError('No density-only additions exercised')
    if 'FULL_CHECKPOINT_RESTORED completed=40 next=41' not in (dirs[1]/'train.log').read_text(): raise ValueError('Density-only resume missing')
    if any((d/'coverage_loss.jsonl').exists() for d in dirs): raise ValueError('Mask loss must stay disabled')
    for d in dirs:
        base.render(d,repo,deadline,split='validation',tags=['best'],smoke=True)
        old.coverage_scores(d,'validation',['best'])
    engine.write_json(REPORTS/'preflight.json',{'status':'passed','flags_off_steps':20,'steps_per_new_arm':100,
        'affine_gradients':affine,'density_events':events,'density_resume_from':40,
        'mask_loss_logs_absent':True,'validation_png_parity':True,'completed_utc':engine.now(),
        'source_freeze_sha256':engine.sha256(REPORTS/'source_freeze.json')})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args = parser.parse_args()
    REPORTS.mkdir(parents=True,exist_ok=True); RUNS.mkdir(parents=True,exist_ok=True)
    old.assert_frozen(old.HERE/'code'); repo = prepare()
    if (REPORTS/'source_freeze.json').exists(): assert_frozen(repo)
    else: engine.write_json(REPORTS/'source_freeze.json',freeze(repo))
    deadline = datetime.fromisoformat(engine.read_json(old.RUNS/'controller.json')['deadline_utc'])
    wire(repo); os.environ['PIDG_EXPORT_ALPHA'] = '1'
    for n in NAMES:
        c,t = configuration(n); engine.write_json(REPORTS/'configs'/(n+'.json'),c)
        engine.write_text_atomic(REPORTS/'configs'/(n+'.py'),t); initialize(n)
    engine.write_json(REPORTS/'plan.json',{'new_arms':{n:configuration(n)[0] for n in NAMES},
        'training_iterations_per_arm':50000,'initialization':'identical random iteration0; no pretrained parent',
        'initial_checkpoint_sha256':engine.read_json(ROOT/'experiments/v3_iso_scratch/initialization.json')['sha256'],
        'split':[120,30,50],'dependency':str(old.RUNS/'controller.json'),'deadline_utc':deadline.isoformat(),
        'preflight':'automatic after dependency completes, before long training','selection':'validation only; lock new arms before test',
        'comparisons':['C4-C0: independent affine effect','C5-C0: independent density effect','C1-C0: independent mask effect'],
        'object_scope':'unsegmented multi-object spatial neighbors; no identity isolation',
        'density_mask_use':'train-only alpha for addition selection, L_mask=0','test_tags':base.TAGS})
    state = {'pid':os.getpid(),'status':'waiting_dependency','started_utc':engine.now(),
        'deadline_utc':deadline.isoformat(),'completed':[],'dependency':str(old.RUNS/'controller.json')}
    stop = threading.Event()
    def watcher():
        while not stop.wait(5):
            try: report()
            except (OSError,KeyError,ValueError): pass
    with engine.lock():
        if (RUNS/'stop_queue.request').exists(): raise RuntimeError('Pause request remains')
        engine.write_json(RUNS/'controller.json',state); report()
        thread = threading.Thread(target=watcher,daemon=True); thread.start()
        try:
            while True:
                if (RUNS/'stop_queue.request').exists(): raise SystemExit(75)
                if datetime.now(timezone.utc)>=deadline: raise SystemExit(75)
                dep = engine.read_json(old.RUNS/'controller.json')
                if dep['status']=='completed' and not engine.alive(dep['pid']): break
                time.sleep(5)
            state['status']='preflight'; engine.write_json(RUNS/'controller.json',state)
            if not (REPORTS/'preflight.json').exists(): preflight(repo,deadline)
            proof = engine.read_json(REPORTS/'preflight.json')
            if proof['status']!='passed' or proof['source_freeze_sha256']!=engine.sha256(REPORTS/'source_freeze.json'):
                raise ValueError('Current-source GPU preflight required')
            if args.smoke: state['status']='preflight_passed'
            else:
                state['status']='training'; engine.write_json(RUNS/'controller.json',state)
                directories = []
                for name in NAMES:
                    d = initialize(name); old.clone_zero(d); directories.append(d)
                    if step(d)<50000 and base.train(d,repo,deadline)==75: raise SystemExit(75)
                    state['completed'].append(name); engine.write_json(RUNS/'controller.json',state)
                selections = {d.parent.name:engine.read_json(d/'metric_best.json') for d in directories}
                lock = RUNS/'selection_lock.json'
                if lock.exists():
                    if engine.read_json(lock)['validation_selections']!=selections: raise ValueError('Locked selections changed')
                else: engine.write_json(lock,{'locked_utc':engine.now(),'validation_selections':selections,'tags':base.TAGS})
                state['status']='final_test'; engine.write_json(RUNS/'controller.json',state)
                for d in directories: base.render(d,repo,deadline); base.pairs(d); old.coverage_scores(d)
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
