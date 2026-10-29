"""Matched C1 / fixed-E / learned-E total50k queue, validation-only selection."""
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
from variants.color_repair import run as parent
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from variants.centroid_tuning import run as coordinator
from variants.coverage_tuning import run as coverage
from variants.c1_material.prepare import prepare

HERE=Path(__file__).parent
RUNS=ROOT/'runs/pidg_c1_material'
REPORTS=ROOT/'experiments/c1_material'
NAMES=['P0_C1_control','P1_fixed_E','P2_learn_E']


def configuration(name,smoke=False,off=False):
    c,text=parent.configuration('C1_repaired',smoke,off)
    active=name in NAMES[1:] and not off
    c.update(material_enabled=active,learn_young=name==NAMES[2] and not off,
        ack_global_effective_material=active,young_init=1.,young_lower=.05,young_upper=20.,poisson_fixed=.3,
        material_weight=1e-4,material_prior_weight=1e-3,material_lr=1e-4,
        material_start=20 if smoke else 3000,material_ramp=10 if smoke else 2000,
        material_interval=10 if smoke else 100,material_neighbors=16,material_samples=32 if smoke else 256,
        material_graph_refresh=20 if smoke else 1000,material_profile_interval=20 if smoke else 1000,
        material_min_J=1e-6)
    return c,text


def source_prefix(smoke=False):
    return parent.RUNS/('_smoke' if smoke else '')/'shared_prefix/rubber_duck'


def freeze(repo):
    prefixes=[source_prefix(s) for s in (False,True)]
    files=[*HERE.glob('*.py'),HERE/'launch.ps1',ROOT/'tests/test_c1_material.py',ROOT/'tests/test_c1_material_runtime.py',
           *list((ROOT/'pidg_ext').glob('*.py')),ROOT/'scripts/export_color_repair_images.py',
           ROOT/'scripts/export_color_repair_four_panel.py']
    for d in prefixes:
        files += [d/p for p in ('model_config.py','metric_config.json','checkpoints/resume.json','checkpoints/resume.metrics.json')]
    return {'parent':parent.freeze(parent.HERE/'code'),'code':engine.runtime_hashes(repo),
        'helpers_and_prefix_receipts':{str(p.relative_to(ROOT)):engine.sha256(p) for p in files}}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json')!=freeze(repo):raise ValueError('C1 material source or prefixes changed')


def wire(repo):
    base.RUNS,base.REPORTS,base.assert_frozen=RUNS,REPORTS,assert_frozen
    engine.RUNS,engine.REPORTS,engine.repo_global,engine.assert_frozen=RUNS,REPORTS,repo,assert_frozen
    coordinator.RUNS,coordinator.REPORTS,coordinator.BASE=RUNS,REPORTS,'v3'
    coordinator.configuration=configuration


def initialize(name,smoke=False,off=False):return coordinator.initialize(name,smoke,off)
def step(directory):return coordinator.step(directory)


def branch(destination,smoke=False):
    source=source_prefix(smoke);boundary=20 if smoke else 3000
    if (destination/'branch.json').exists():return
    if step(destination)>=0:raise ValueError('Unreceipted destination checkpoint')
    if step(source)!=boundary:raise ValueError('Wrong pre-material prefix iteration')
    if (source/'model_config.py').read_text()!=(destination/'model_config.py').read_text():
        raise ValueError('Control optimizer recipe changed')
    manifest=engine.read_json(source/'checkpoints/resume.json')
    if engine.sha256(source/'checkpoints/resume.pt')!=manifest['sha256']:raise ValueError('Invalid source prefix')
    for folder in ('point_cloud','deform','checkpoints'):shutil.copytree(source/folder,destination/folder,dirs_exist_ok=True)
    for filename in ('metric_best.json','metric_history.jsonl'):shutil.copyfile(source/filename,destination/filename)
    sidecar=destination/'checkpoints/resume.metrics.json';saved=engine.read_json(sidecar)
    original=engine.read_json(source/'metric_config.json');current=engine.read_json(destination/'metric_config.json')
    differences={k for k in set(original)|set(current) if original.get(k)!=current.get(k)}
    allowed={'coverage_density','ack_global_effective_material','young_init','young_lower','young_upper','poisson_fixed',
        'material_enabled','learn_young','material_weight','material_prior_weight','material_lr','material_start',
        'material_ramp','material_interval','material_neighbors','material_samples','material_graph_refresh',
        'material_profile_interval','material_min_J'}
    if not differences<=allowed or saved['config']!=original:raise ValueError('Undeclared prefix differences')
    saved['config']=current;engine.write_json(sidecar,saved)
    engine.write_json(destination/'branch.json',{'source':str(source),'destination':str(destination),
        'iteration':boundary,'identical_full_checkpoint_sha256':manifest['sha256'],
        'initialize_material_at_branch':True,'source_config':original,'branch_config':current,
        'initialization':'existing random-step0 C1 followed by common pre-material prefix; not continuation of final50k',
        'changed_flags':sorted(differences),'created_utc':engine.now()})


def train_arms(repo,deadline,smoke=False):
    target=100 if smoke else 50000
    arms=[]
    for name in NAMES:
        d=initialize(name,smoke);branch(d,smoke);arms.append(d)
        if smoke and name==NAMES[2] and step(d)<70:
            if base.train(d,repo,deadline,True,limit=70)!=75:raise ValueError('Material resume boundary failed')
        if step(d)<target and base.train(d,repo,deadline,smoke)==75:raise SystemExit(75)
    return arms


def report():
    state=engine.read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {}
    active=engine.read_json(RUNS/'active.json') if (RUNS/'active.json').exists() else {}
    rows=[]
    for name in NAMES:
        d=RUNS/name/'rubber_duck';h=base.history(d)
        log=d/'material_loss.jsonl';e=None
        if log.exists():
            records=[json.loads(s) for s in engine.tail(log,20000).splitlines() if s.startswith('{')]
            if records:e=records[-1]
        rows.append({'id':name,'iteration':base.progress(d),'history':h,'material':e,
            'test':engine.read_json(d/'results.json').get('ours_50000') if (d/'results.json').exists() else None})
    result={'controller':state,'active':active,'updated_utc':engine.now(),'rows':rows,
        'scope':'one shared global effective stiffness prior, not true multi-material recovery',
        'data_identifiability':'not_established','units':'normalized effective stiffness, not Pa'}
    engine.write_json(REPORTS/'results.json',result)
    lines=['# C1 / fixed-E / learned-E total50k comparison','',
        'Status: '+state.get('status','preparing')+' | Active: '+active.get('id','-')+' / '+active.get('phase','-'),'',
        'Three matched full3k branches, total50k each;120 train /30 validation /50 test. Historical test exposure persists.',
        'P0: C1 control. P1: same C1 + fixed E=1. P2: same constitutive objective + learned shared E.',
        'Fixed Poisson0.3; E bounded[0.05,20]. Global effective prior, no true material labels, no calibrated Pa.',
        'Mean constitutive weight1e-4, interval100 with compensation; start3k/ramp2k. Prior1e-3 on log(E/E0)^2.',
        'Only validation selects snapshots; final test opens after all three finish and selections lock.','',
        '| Model | Iteration | Latest val PSNR | E | Test PSNR | SSIM | LPIPS | MS-SSIM |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        val=row['history'][-1]['metrics']['PSNR'] if row['history'] else '-'
        values=row['test'] or {};e=row['material']['E'] if row['material'] else '-'
        lines.append('| '+' | '.join([row['id'],str(row['iteration'])+'/50000',str(val),str(e),
            *[f'{values[k]:.7f}' if k in values else '-' for k in engine.METRICS]])+' |')
    if all(row['test'] for row in rows):
        lines+=['','Final50k deltas versus P0:']
        for row in rows[1:]:lines.append(row['id']+': '+json.dumps({k:row['test'][k]-rows[0]['test'][k] for k in engine.METRICS}))
    lines+=['','[5-second monitor](monitor.html) | [Plan](plan.json) | [Preflight](preflight.json)',
        '', 'GT/render galleries: '+str(RUNS/'render_comparisons'),'',
        'Conditional E profiles freeze neural fields; they do not prove data identifiability.',
        'Updated UTC: '+result['updated_utc']]
    engine.write_text_atomic(REPORTS/'results.md','\n'.join(lines)+'\n')
    page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>C1 material comparison</title><style>body{font:16px system-ui;background:#111c2c;color:#edf5ff;margin:25px}section,svg{background:#203047;padding:14px;margin:12px 0}progress{width:85%}a{color:#8de4b1}svg{max-width:100%}</style><h1>C1 / fixed E / learned E</h1><p>'+html.escape(state.get('status','preparing')+' | '+active.get('id','-')+' / '+active.get('phase','-'))+'</p><p>Shared3k state; total50k each;120/30/50. Global effective material prior; E is not calibrated Pa.</p>'
    for row in rows:
        e='-' if row['material'] is None else f'{row["material"]["E"]:.6f}'
        page+=f'<section><b>{row["id"]}</b><h2>{row["iteration"]:,} /50,000 iter</h2><progress value="{row["iteration"]}" max="50000"></progress><p>E: {e}</p></section>'
    for key in engine.METRICS:
        values=[h['metrics'][key] for row in rows for h in row['history']]
        if not values:continue
        lo,hi=min(values),max(values);span=max(hi-lo,1e-8)
        page+=f'<h2>Validation {key}</h2><svg viewBox="0 0 900 280" width="900"><text x="5" y="20" fill="white">{hi:.5f}</text><text x="5" y="220" fill="white">{lo:.5f}</text>'
        for i,(row,color) in enumerate(zip(rows,['#86d1ff','#8de4b1','#ffbe83'])):
            points=' '.join(f'{80+790*h["iteration"]/50000:.1f},{220-190*(h["metrics"][key]-lo)/span:.1f}' for h in row['history'])
            page+=f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="{80+i*260}" y="255" fill="{color}">{row["id"]}</text>'
        page+='</svg>'
    engine.write_text_atomic(REPORTS/'monitor.html',page+'<p><a href="results.md">Results and protocol</a></p>')


def preflight(repo,deadline):
    environment=os.environ.copy();environment.update(PIDG_COLOR_REPAIR_GPU_TEST='1',
        PIDG_HASH_SOURCE_DIR=str(engine.SOURCE/'hashencoder'),PIDG_HASH_BUILD_DIR=str(engine.SOURCE/'tmp_build'))
    gpu=subprocess.run([sys.executable,'-m','pytest','tests/test_color_repair_gpu.py','-q'],cwd=ROOT,
        env=environment,capture_output=True,text=True)
    engine.write_text_atomic(REPORTS/'gpu_test.log',gpu.stdout+gpu.stderr)
    if gpu.returncode:raise ValueError('Inherited C1 renderer CUDA tests failed')
    off=initialize('flags_off',True,True)
    if step(off)<20 and base.train(off,repo,deadline,True,off=True)!=0:raise ValueError('Flags-off smoke failed')
    arms=train_arms(repo,deadline,True)
    if (arms[0]/'material_loss.jsonl').exists():raise ValueError('Control unexpectedly used new physics')
    logs=[ [json.loads(s) for s in (d/'material_loss.jsonl').read_text().splitlines()] for d in arms[1:] ]
    for rows in logs:
        if not any(r['F_gradient_norm']>0 and r['valid_F_fraction']>0 for r in rows):
            raise ValueError('Constitutive loss did not reach valid trajectory F')
    if any(abs(r['E']-1)>1e-10 for r in logs[0]):raise ValueError('Fixed E changed')
    if not any(abs(r['E']-1)>1e-7 for r in logs[1]):raise ValueError('Learned E did not update')
    if 'FULL_CHECKPOINT_RESTORED completed=70 next=71' not in (arms[2]/'train.log').read_text():
        raise ValueError('Learned material resume was not exercised')
    for d in arms:base.render(d,repo,deadline,split='validation',tags=['best'],smoke=True)
    engine.write_json(REPORTS/'preflight.json',{'status':'passed','flags_off_steps':20,'smoke_each':100,
        'learned_resume_from':70,'fixed_E_unchanged':True,'learned_E_updated':True,'F_gradients_finite_nonzero':True,
        'validation_PNG_parity':True,'source_freeze_sha256':engine.sha256(REPORTS/'source_freeze.json'),
        'completed_utc':engine.now()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--hours',type=float,default=24.)
    args=parser.parse_args()
    if args.hours<=0:raise ValueError('Positive budget required')
    REPORTS.mkdir(parents=True,exist_ok=True);RUNS.mkdir(parents=True,exist_ok=True)
    parent.assert_frozen(parent.HERE/'code');repo=prepare()
    if (REPORTS/'source_freeze.json').exists():assert_frozen(repo)
    else:engine.write_json(REPORTS/'source_freeze.json',freeze(repo))
    wire(repo)
    plan={'arms':{n:configuration(n)[0] for n in NAMES},'split':[120,30,50],'total_iterations_each':50000,
        'seed':0,'shared_prefix':3000,'source_prefix':str(source_prefix()),'budget_hours':args.hours,
        'scope':'single shared effective global material prior, no material labels; not true multi-material recovery',
        'units':'normalized effective E; not calibrated Pa','material_law':'existing compressible Neo-Hookean',
        'kinematic_consistency':False,'no_extra_training_after50k':True,'historical_test_exposure':True,
        'selection':'validation-only; lock all3 before final test','test_tags':base.TAGS,
        'conditional_profiles_are_not_global_identifiability':True}
    if (REPORTS/'plan.json').exists() and engine.read_json(REPORTS/'plan.json')!=plan:raise ValueError('Immutable material plan changed')
    engine.write_json(REPORTS/'plan.json',plan)
    for n in NAMES:
        c,text=configuration(n);engine.write_json(REPORTS/'configs'/(n+'.json'),c);engine.write_text_atomic(REPORTS/'configs'/(n+'.py'),text)
    deadline=datetime.now(timezone.utc)+timedelta(hours=args.hours)
    state={'pid':os.getpid(),'status':'preflight','mode':'smoke' if args.smoke else 'long',
        'started_utc':engine.now(),'deadline_utc':deadline.isoformat(),'completed':[]}
    stop=threading.Event()
    def monitor():
        while not stop.wait(5):
            try:report()
            except (OSError,ValueError,KeyError):pass
    with engine.lock():
        if (RUNS/'stop_queue.request').exists():raise ValueError('Pause request remains')
        engine.write_json(RUNS/'controller.json',state);report();thread=threading.Thread(target=monitor,daemon=True);thread.start()
        os.environ['PIDG_EXPORT_ALPHA']='1'
        try:
            proof=engine.read_json(REPORTS/'preflight.json') if (REPORTS/'preflight.json').exists() else {}
            if proof.get('source_freeze_sha256')!=engine.sha256(REPORTS/'source_freeze.json'):preflight(repo,deadline)
            if args.smoke:state['status']='preflight_passed'
            else:
                state['status']='training';engine.write_json(RUNS/'controller.json',state)
                arms=train_arms(repo,deadline)
                selection={d.parent.name:engine.read_json(d/'metric_best.json') for d in arms}
                engine.write_json(RUNS/'selection_lock.json',{'locked_utc':engine.now(),'validation_selections':selection,'tags':base.TAGS})
                state.update(status='final_test',completed=NAMES.copy());engine.write_json(RUNS/'controller.json',state)
                for d in arms:base.render(d,repo,deadline);base.pairs(d);coverage.coverage_scores(d)
                from variants.c1_material.export import export
                export()
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
