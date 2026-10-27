"""Three scratch50k repair packages; shared3k and exact pre-density15k branches."""
import argparse
from datetime import datetime,timedelta,timezone
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from variants.coverage_tuning import run as old
from variants.centroid_tuning import run as coordinator
from variants.v3_iso_scratch import run as scratch
from variants.color_repair.prepare import prepare
from variants.coverage_tuning.data import audit_training

HERE=Path(__file__).resolve().parent
RUNS=ROOT/'runs/pidg_color_repair'
REPORTS=ROOT/'experiments/color_repair'
NAMES=['C1_repaired','C2_repaired','C3_repaired']


def configuration(name,smoke=False,off=False):
    c,text=old.configuration('C3_mask_affine_density',smoke,off)
    affine=name in (NAMES[1],NAMES[2],'affine_prefix')
    density=name in (NAMES[2],'shared_prefix','affine_prefix')
    c.pop('single_object',None)
    c.update(positive_color=not off,color_beta=5.,mask_weight=0. if off else .03,
        mask_start=20 if smoke else 3000,mask_ramp=10 if smoke else 2000,
        foreground_rgb_weight=0. if off else .1,affine_weight=5e-5 if affine and not off else 0.,
        affine_radius=.1 if smoke else .05,affine_start=20 if smoke else 3000,
        affine_ramp=10 if smoke else 2000,ack_unsegmented_affine=affine and not off,
        coverage_density=density and not off,coverage_start=40 if smoke else 16000,
        coverage_until=100 if smoke else 36000,coverage_window=20 if smoke else 2000,
        coverage_add_limit=8 if smoke else 256,clone_color_std_max=1. if smoke else .15,
        centroid_weight=0.)
    return c,text


def freeze(repo):
    return {'parent':old.freeze(old.HERE/'code'),'code':engine.runtime_hashes(repo),
        'inputs':{str(p.relative_to(ROOT)):engine.sha256(p) for p in [*HERE.glob('*.py'),HERE/'launch.ps1',
            ROOT/'tests/test_color_repair.py',ROOT/'tests/test_color_repair_gpu.py']},
        'alpha_audit_sha256':engine.sha256(REPORTS/'alpha_audit.json'),
        'diagnostic_sha256':engine.sha256(REPORTS/'original_color_diagnostic.json')}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json')!=freeze(repo):raise ValueError('Repair source/data changed')


def wire(repo):
    base.RUNS,base.REPORTS,base.assert_frozen=RUNS,REPORTS,assert_frozen
    engine.RUNS,engine.REPORTS,engine.repo_global,engine.assert_frozen=RUNS,REPORTS,repo,assert_frozen
    coordinator.RUNS,coordinator.REPORTS,coordinator.BASE=RUNS,REPORTS,'v3'
    coordinator.configuration=configuration


def initialize(name,smoke=False,off=False):return coordinator.initialize(name,smoke,off)
def step(directory):return coordinator.step(directory)


def branch(source,destination,boundary):
    receipt=destination/'branch.json'
    if receipt.exists():
        previous=engine.read_json(receipt)
        if previous['source']!=str(source) or previous['iteration']!=boundary:raise ValueError('Wrong repair branch')
        return
    if (destination/'checkpoints/resume.pt').exists():raise ValueError('Unreceipted destination checkpoint')
    a,b=[engine.read_json(d/'metric_config.json') for d in (source,destination)]
    changed={key for key in set(a)|set(b) if a.get(key)!=b.get(key)}
    if not changed<={'affine_weight','coverage_density','ack_unsegmented_affine'}:
        raise ValueError('Undeclared configuration changes at repair branch')
    if (source/'model_config.py').read_text()!=(destination/'model_config.py').read_text() or step(source)!=boundary:
        raise ValueError('Repair optimizer/branch boundary changed')
    for folder in ('point_cloud','deform','checkpoints'):shutil.copytree(source/folder,destination/folder,dirs_exist_ok=True)
    for filename in ('metric_best.json','metric_history.jsonl'):shutil.copyfile(source/filename,destination/filename)
    checkpoint=destination/'checkpoints/resume.pt';digest=engine.sha256(checkpoint)
    if digest!=engine.read_json(source/'checkpoints/resume.json')['sha256']:raise ValueError('Full branch state changed')
    sidecar=checkpoint.with_suffix('.metrics.json');state=engine.read_json(sidecar)
    if state['config']!=a:raise ValueError('Wrong checkpoint-sidecar config')
    state['config']=b;engine.write_json(sidecar,state)
    engine.write_json(receipt,{'source':str(source),'destination':str(destination),'iteration':boundary,
        'identical_full_checkpoint_sha256':digest,'source_config':a,'branch_config':b,
        'changed_component_flags':sorted(changed),'created_utc':engine.now()})


def shared_prefix(repo,deadline,smoke=False):
    d=initialize('shared_prefix',smoke);boundary=20 if smoke else 3000
    if step(d)<0:
        os.environ['PIDG_CAPTURE_INITIAL_ONLY']='1'
        try:
            if base.train(d,repo,deadline,smoke)!=75:raise ValueError('Initial capture failed')
        finally:os.environ.pop('PIDG_CAPTURE_INITIAL_ONLY',None)
        from reproduction.scripts.training_checkpoint import load_state
        scratch.validate_initial_state(load_state(d/'checkpoints/resume.pt'))
        engine.write_json(REPORTS/('smoke_initialization.json' if smoke else 'initialization.json'),
            {'iteration':0,'optimizer_updates':0,'pretrained_model_used':False,'sha256':engine.sha256(d/'checkpoints/resume.pt'),
             'color_initialization':engine.read_json(d/'initial_color_conversion.json')})
    if step(d)<boundary and base.train(d,repo,deadline,smoke,limit=boundary)!=75:raise ValueError('Common prefix stop failed')
    if step(d)!=boundary:raise ValueError('Common prefix must remain frozen')
    return d


def train_arms(repo,deadline,smoke=False):
    common=shared_prefix(repo,deadline,smoke)
    first=initialize(NAMES[0],smoke);branch(common,first,20 if smoke else 3000)
    target=100 if smoke else 50000
    if step(first)<target and base.train(first,repo,deadline,smoke)==75:raise SystemExit(75)
    middle=initialize('affine_prefix',smoke);branch(common,middle,20 if smoke else 3000)
    boundary=40 if smoke else 15000
    if step(middle)<boundary and base.train(middle,repo,deadline,smoke,limit=boundary)!=75:raise ValueError('Affine prefix did not stop')
    if step(middle)!=boundary:raise ValueError('Wrong shared pre-density state')
    arms=[first]
    for name in NAMES[1:]:
        d=initialize(name,smoke);branch(middle,d,boundary);arms.append(d)
        if smoke and name==NAMES[2] and step(d)<70:
            if base.train(d,repo,deadline,True,limit=70)!=75:raise ValueError('Density resume smoke boundary failed')
        if step(d)<target and base.train(d,repo,deadline,smoke)==75:raise SystemExit(75)
    return arms


def report():
    state=engine.read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {}
    active=engine.read_json(RUNS/'active.json') if (RUNS/'active.json').exists() else {}
    rows=[]
    for name in ['shared_prefix','affine_prefix',*NAMES]:
        d=RUNS/name/'rubber_duck';history=base.history(d)
        rows.append({'id':name,'iteration':base.progress(d),'history':history,
            'test':engine.read_json(d/'results.json').get('ours_50000') if (d/'results.json').exists() else None,
            'coverage':engine.read_json(d/'coverage_metrics_test.json').get('50000',{}).get('mean') if (d/'coverage_metrics_test.json').exists() else None})
    comparisons=[]
    for row,oldname in zip(rows[2:],old.NAMES[1:]):
        previous=engine.read_json(old.RUNS/oldname/'rubber_duck/results.json')['ours_50000']
        if row['test']:comparisons.append({'id':row['id'],'historical':oldname,
            'final50k_delta':{key:row['test'][key]-previous[key] for key in engine.METRICS}})
    result={'updated_utc':engine.now(),'controller':state,'active':active,'rows':rows,'versus_historical':comparisons,
        'scope':'Combined color/mask/foreground repair packages; not a single-change causal ablation versus historical runs',
        'split':[120,30,50],'no_pretrained_parent':True,'shared_initial_prefix':3000,'shared_C2_C3_prefix':15000,
        'limitations':['unsegmented multi-object neighbors','single scene and seed','historical test exposure','CUDA nondeterminism after branch']}
    engine.write_json(REPORTS/'results.json',result)
    lines=['# C1 / C2 / C3 color-repair retraining','',
        'Status: '+state.get('status','preparing')+' | Active: '+active.get('id','-')+' / '+active.get('phase','-'),'',
        'Each arm: random initialization,50k total. Shared3k; C2/C3 share full15k pre-density state.',
        'V3 image objective and shared time retained. Positive SH radiance; inverse initial DC; delayed/ramped mask0.03; foreground RGB0.1.',
        'C2/C3 affine5e-5 within radius0.05; no claimed object-ID isolation. C3 uses training RGB-consistent clones, at most256/window.',
        'Only120 training RGB/alpha targets enter losses;30 validation selects four snapshots;50 final test stays locked until all complete.', '',
        '| Stage | Iteration | Validation PSNR | Test PSNR | SSIM | LPIPS | MS-SSIM |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values=row['test'] or {};val=row['history'][-1]['metrics']['PSNR'] if row['history'] else '-'
        target=3000 if row['id']=='shared_prefix' else 15000 if row['id']=='affine_prefix' else 50000
        lines.append('| '+' | '.join([row['id'],f"{row['iteration']}/{target}",str(val),*[f'{values[k]:.7f}' if k in values else '-' for k in engine.METRICS]])+' |')
    for c in comparisons:lines+=['',c['id']+' minus historical '+c['historical']+': '+json.dumps(c['final50k_delta'])]
    lines+=['','These are repair-package comparisons, not attribution to one modified component.',
        '[5-second charts](monitor.html) | [Plan](plan.json) | [Original color diagnostic](original_color_diagnostic.json) | [Preflight](preflight.json)',
        '', 'GT/render galleries: '+str(RUNS/'render_comparisons'),'','Updated UTC: '+result['updated_utc']]
    engine.write_text_atomic(REPORTS/'results.md','\n'.join(lines)+'\n')
    page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>C1-C3 color repair</title><style>body{font:16px system-ui;background:#111c2c;color:#edf5ff;margin:30px}section,svg{background:#203047;padding:12px;margin:12px 0}progress{width:85%}svg{max-width:100%}a{color:#8be2b6}</style><h1>C1 / C2 / C3 color repair</h1><p>'+html.escape(state.get('status','preparing')+' | '+active.get('id','-')+' / '+active.get('phase','-'))+'</p><p>50k total each; same3k prefix and C2/C3 same15k state. Refresh5 seconds.</p>'
    for row in rows:
        target=3000 if row['id']=='shared_prefix' else 15000 if row['id']=='affine_prefix' else 50000
        page+=f"<section><b>{row['id']}</b><h2>{row['iteration']:,} / {target:,} iter</h2><progress value='{row['iteration']}' max='{target}'></progress></section>"
    for key in engine.METRICS:
        values=[h['metrics'][key] for row in rows[2:] for h in row['history']]
        if not values:continue
        lo,hi=min(values),max(values);span=max(hi-lo,1e-8)
        page+=f'<h2>Validation {key}</h2><svg viewBox="0 0 900 280" width="900"><text x="5" y="20" fill="white">{hi:.5f}</text><text x="5" y="220" fill="white">{lo:.5f}</text>'
        for i,(row,color) in enumerate(zip(rows[2:],['#86d1ff','#8de4b1','#ffbe83'])):
            points=' '.join(f"{80+790*h['iteration']/50000:.1f},{220-190*(h['metrics'][key]-lo)/span:.1f}" for h in row['history'])
            page+=f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="{80+i*260}" y="255" fill="{color}">{row["id"]}</text>'
        page+='</svg>'
    engine.write_text_atomic(REPORTS/'monitor.html',page+'<p><a href="results.md">Results and protocol</a></p>')


def preflight(repo,deadline):
    environment=os.environ.copy();environment.update(PIDG_COLOR_REPAIR_GPU_TEST='1',
        PIDG_HASH_SOURCE_DIR=str(engine.SOURCE/'hashencoder'),PIDG_HASH_BUILD_DIR=str(engine.SOURCE/'tmp_build'))
    result=subprocess.run([sys.executable,'-m','pytest','tests/test_color_repair_gpu.py','-q'],cwd=ROOT,
        env=environment,capture_output=True,text=True)
    engine.write_text_atomic(REPORTS/'gpu_test.log',result.stdout+result.stderr)
    if result.returncode:raise ValueError('Color-repair CUDA tests failed')
    off=initialize('flags_off',True,True)
    if step(off)<20 and base.train(off,repo,deadline,True,off=True)!=0:raise ValueError('Flags-off short train failed')
    arms=train_arms(repo,deadline,True)
    events=[json.loads(line) for line in (arms[2]/'coverage_density.jsonl').read_text().splitlines()]
    if not any(row['added']>0 for row in events):raise ValueError('Smoke did not exercise color-aware additions')
    if 'FULL_CHECKPOINT_RESTORED completed=70 next=71' not in (arms[2]/'train.log').read_text():
        raise ValueError('Color-evidence resume was not exercised')
    for d in arms:
        if not (d/'repair_loss.jsonl').exists():raise ValueError('Missing repaired mask/foreground loss')
        base.render(d,repo,deadline,split='validation',tags=['best'],smoke=True)
    engine.write_json(REPORTS/'preflight.json',{'status':'passed','flags_off_steps':20,'smoke_steps_each':100,
        'real_cuda_tests':4,'shared_prefix':20,'shared_affine_prefix':40,'density_resume_from':70,
        'color_aware_density_events':events,'validation_png_parity':True,
        'source_freeze_sha256':engine.sha256(REPORTS/'source_freeze.json'),'completed_utc':engine.now()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--hours',type=float,default=24.)
    args=parser.parse_args()
    if args.hours<=0:raise ValueError('Positive budget required')
    RUNS.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    if not (REPORTS/'original_color_diagnostic.json').exists():raise ValueError('Read-only original color diagnosis required first')
    old.assert_frozen(old.HERE/'code');repo=prepare()
    if not (REPORTS/'alpha_audit.json').exists():engine.write_json(REPORTS/'alpha_audit.json',audit_training(base.DATA/'training'))
    if (REPORTS/'source_freeze.json').exists():assert_frozen(repo)
    else:engine.write_json(REPORTS/'source_freeze.json',freeze(repo))
    wire(repo)
    plan={'arms':{n:configuration(n)[0] for n in NAMES},'seed':0,'split':[120,30,50],'total_iterations_each':50000,
        'source':'new random step0; V3 model/loss/shared clock; historical checkpoints only for diagnostics and comparison',
        'shared_prefix_iterations':3000,'C2_C3_shared_pre_density_iterations':15000,
        'repair_package':'softplus beta5 with inverse initial DC; mask0.03 after3k/ramp2k; foreground RGB0.1; affine5e-5 radius0.05; color-consistent fresh clone DC',
        'selection':'validation only; lock all three before final test','test_tags':base.TAGS,
        'budget_hours':args.hours,'historical_test_exposure':True,'object_id_isolation':False,
        'interpretation':'Package-level historical comparison; internal C2/C3 share exact15k full state.'}
    if (REPORTS/'plan.json').exists() and engine.read_json(REPORTS/'plan.json')!=plan:raise ValueError('Immutable repair plan changed')
    engine.write_json(REPORTS/'plan.json',plan)
    for n in NAMES:
        c,text=configuration(n);engine.write_json(REPORTS/'configs'/(n+'.json'),c);engine.write_text_atomic(REPORTS/'configs'/(n+'.py'),text)
    deadline=datetime.now(timezone.utc)+timedelta(hours=args.hours)
    state={'pid':os.getpid(),'status':'preflight','started_utc':engine.now(),'deadline_utc':deadline.isoformat(),
        'mode':'smoke' if args.smoke else 'long','completed':[]}
    stop=threading.Event()
    def watch():
        while not stop.wait(5):
            try:report()
            except (OSError,KeyError,ValueError):pass
    with engine.lock():
        if (RUNS/'stop_queue.request').exists():raise ValueError('User pause request remains')
        engine.write_json(RUNS/'controller.json',state);report();thread=threading.Thread(target=watch,daemon=True);thread.start()
        os.environ['PIDG_EXPORT_ALPHA']='1'
        try:
            proof=engine.read_json(REPORTS/'preflight.json') if (REPORTS/'preflight.json').exists() else {}
            if proof.get('source_freeze_sha256')!=engine.sha256(REPORTS/'source_freeze.json'):preflight(repo,deadline)
            if args.smoke:state['status']='preflight_passed'
            else:
                state['status']='training';engine.write_json(RUNS/'controller.json',state)
                arms=train_arms(repo,deadline)
                selections={d.parent.name:engine.read_json(d/'metric_best.json') for d in arms}
                lock=RUNS/'selection_lock.json'
                if lock.exists() and engine.read_json(lock)['validation_selections']!=selections:raise ValueError('Locked selections changed')
                engine.write_json(lock,{'locked_utc':engine.now(),'validation_selections':selections,'tags':base.TAGS})
                state['completed']=NAMES.copy();state['status']='final_test';engine.write_json(RUNS/'controller.json',state)
                for d in arms:base.render(d,repo,deadline);base.pairs(d);old.coverage_scores(d)
                state['status']='completed'
            assert_frozen(repo)
        except SystemExit as error:
            state['status']='saved_stopped' if error.code==75 else 'failed'
            if error.code!=75:raise
        except Exception as error:
            state.update(status='failed',error=str(error));raise
        finally:
            state['finished_utc']=engine.now();engine.write_json(RUNS/'controller.json',state)
            stop.set();thread.join(timeout=6);report()


if __name__=='__main__':main()
