"""Matched C1 / frozen identity / self-supervised identity 50k comparison."""
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

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.color_repair import run as parent
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from variants.coverage_tuning import run as coverage
from variants.centroid_tuning import run as coordinator
from .prepare import prepare

HERE=Path(__file__).parent;RUNS=ROOT/'runs/pidg_object_id';REPORTS=ROOT/'experiments/object_id'
NAMES=['O0_C1_control','O1_frozen_ID','O2_selfsup_ID']


def configuration(name,smoke=False,off=False):
    c,text=parent.configuration('C1_repaired',smoke,off)
    c.update(object_id_enabled=name in NAMES[1:] and not off,
        learn_object_identity=name==NAMES[2] and not off,object_capacity=16,
        object_start=20 if smoke else 3000,object_ramp=20 if smoke else 5000,
        object_blend=.25,object_alignment=1e-3,object_sharpness=5e-5,object_complexity=1e-6,
        object_lr=2e-4,object_total_iterations=100 if smoke else 50000,
        object_audit_interval=20 if smoke else 1000,object_prune_start=60 if smoke else 10000,
        object_prune_interval=20 if smoke else 2000,object_min_mass=.001)
    return c,text


def prefix(smoke=False):
    return RUNS/'_smoke/shared_prefix/rubber_duck' if smoke else parent.RUNS/'shared_prefix/rubber_duck'


def freeze(repo):
    files=[*HERE.glob('*.py'),HERE/'launch.ps1',ROOT/'tests/test_object_id.py',ROOT/'tests/test_object_id_runtime.py',ROOT/'tests/test_object_id_gpu.py']
    files += [prefix()/p for p in ('model_config.py','metric_config.json','checkpoints/resume.json','checkpoints/resume.metrics.json')]
    files += list((base.DATA/'training').glob('*.json'))+list((base.DATA/'final_test').glob('*.json'))
    return {'parent':parent.freeze(parent.HERE/'code'),'code':engine.runtime_hashes(repo),
            'helpers_inputs':{str(p.relative_to(ROOT)):engine.sha256(p) for p in files}}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json')!=freeze(repo):raise ValueError('Object experiment source/data changed')


def wire(repo):
    base.RUNS,base.REPORTS,base.assert_frozen=RUNS,REPORTS,assert_frozen
    engine.RUNS,engine.REPORTS,engine.repo_global,engine.assert_frozen=RUNS,REPORTS,repo,assert_frozen
    coordinator.RUNS,coordinator.REPORTS,coordinator.configuration=RUNS,REPORTS,configuration
    coordinator.BASE='C1_repaired'


def initialize(name,smoke=False,off=False):
    return coordinator.initialize(name,smoke,off)


def branch(source,destination,boundary):
    if (destination/'branch.json').exists():return
    if (destination/'checkpoints/resume.pt').exists():raise ValueError('Unreceipted object branch')
    if coordinator.step(source)!=boundary:raise ValueError('Wrong shared prefix boundary')
    if (source/'model_config.py').read_bytes()!=(destination/'model_config.py').read_bytes():raise ValueError('Optimization recipe changed')
    a,b=[engine.read_json(d/'metric_config.json') for d in (source,destination)]
    changed={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    if any(not k.startswith('object_') and k not in ('learn_object_identity','coverage_density') for k in changed):
        raise ValueError('Undeclared branch configuration changes: '+str(changed))
    for folder in ('point_cloud','deform','checkpoints'):shutil.copytree(source/folder,destination/folder,dirs_exist_ok=True)
    for name in ('metric_best.json','metric_history.jsonl'):shutil.copyfile(source/name,destination/name)
    path=destination/'checkpoints/resume.pt';manifest=engine.read_json(path.with_suffix('.json'))
    if engine.sha256(path)!=manifest['sha256']:raise ValueError('Copied tensor state differs')
    state=engine.read_json(path.with_suffix('.metrics.json'))
    if state['config']!=a:raise ValueError('Unexpected prefix selection state')
    state['config']=b;engine.write_json(path.with_suffix('.metrics.json'),state)
    engine.write_json(destination/'branch.json',{'source':str(source),'destination':str(destination),
        'iteration':boundary,'identical_full_checkpoint_sha256':manifest['sha256'],
        'initialize_object_at_branch':True,'changed_flags':sorted(changed),'created_utc':engine.now()})


def report():
    state=engine.read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {}
    active=engine.read_json(RUNS/'active.json') if (RUNS/'active.json').exists() else {}
    rows=[]
    for name in NAMES:
        d=RUNS/name/'rubber_duck';history=base.history(d)
        object_log=d/'object_loss.jsonl';last=None
        if object_log.exists():
            lines=object_log.read_text(encoding='utf-8').splitlines()
            if lines:last=json.loads(lines[-1])
        rows.append({'id':name,'iteration':base.progress(d),'history':history,'object':last,
            'test':engine.read_json(d/'results.json').get('ours_50000') if (d/'results.json').exists() else None})
    result={'updated_utc':engine.now(),'controller':state,'active':active,'rows':rows,
        'split':[120,30,50],'total_iterations_each':50000,'source_prefix_iteration':3000,
        'object_supervision':'none: no instance masks/IDs, no clustering pseudo-targets',
        'C1_foreground_alpha_supervision_retained':True,'candidate_capacity_not_object_count':16,
        'selection':'validation-only; final test after all3 selections locked',
        'limitations':['single scene/seed','not exact MovingParts/DynaVol reproduction','no ground-truth instance accuracy','historical test exposure']}
    engine.write_json(REPORTS/'results.json',result)
    lines=['# C1 self-supervised Object ID comparison','',
        'Status: '+state.get('status','preparing')+' | Active: '+active.get('id','-')+' / '+active.get('phase','-'),'',
        'MovingParts-inspired shared object motion + retained nonrigid C1 path. No instance labels or pseudo-ID targets.',
        'C1 alpha/RGB/perceptual recipe retained in all arms; not a fully RGB-only experiment.',
        'Matched full3k prefix; total50k each;120 train /30 validation /50 final test.',
        '16 candidate slots are a capacity ceiling, not a known or forced object count. Effective occupancy is a diagnostic.',
        '', '| Model | Current iter | Validation PSNR | Final test PSNR | Effective slots >1% |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        val=r['history'][-1]['metrics']['PSNR'] if r['history'] else '-'
        lines.append(f"| {r['id']} | {r['iteration']}/50000 | {val} | {(r['test'] or {}).get('PSNR','-')} | {(r['object'] or {}).get('effective_slots_1pct','-')} |")
    lines+=['','[Monitor](monitor.html) | [Plan](plan.json) | [Preflight](preflight.json)',
            '','GT/render export after training: '+str(RUNS/'render_comparisons')]
    engine.write_text_atomic(REPORTS/'results.md','\n'.join(lines)+'\n')
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>C1 Object ID</title><style>body{font:16px system-ui;background:#142033;color:#edf5ff;margin:24px}section,svg{background:#203047;padding:12px;margin:12px 0}progress{width:90%}a{color:#9be0b8}</style><h1>C1 / frozen ID / self-supervised ID</h1><p>'+html.escape(state.get('status','preparing')+' | '+active.get('id','-')+' / '+active.get('phase','-'))+'</p><p>Shared3k  ->  total50k each.120/30/50. No instance labels; C1 foreground alpha retained.16 candidate slots do not mean16 objects.</p>'
    for r in rows:
        o=r['object'] or {};page+=f"<section><b>{r['id']}</b><h2>{r['iteration']:,} / 50,000 iter</h2><progress value='{r['iteration']}' max='50000'></progress><p>Effective slots >1%: {o.get('effective_slots_1pct','-')} | RGB -> ID grad: {o.get('RGB_to_identity_gradient_norm','-')}</p></section>"
    for key in engine.METRICS:
        values=[h['metrics'][key] for r in rows for h in r['history']]
        if not values:continue
        lo,hi=min(values),max(values);span=max(hi-lo,1e-8)
        page+=f'<h2>Validation {key}</h2><svg viewBox="0 0 900 280" width="900"><text x="5" y="20" fill="white">{hi:.5f}</text><text x="5" y="220" fill="white">{lo:.5f}</text>'
        for i,(r,color) in enumerate(zip(rows,['#86d1ff','#ffbe83','#8de4b1'])):
            points=' '.join(f'{80+790*h["iteration"]/50000:.1f},{220-190*(h["metrics"][key]-lo)/span:.1f}' for h in r['history'])
            page+=f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="{80+i*260}" y="255" fill="{color}">{r["id"]}</text>'
        page+='</svg>'
    engine.write_text_atomic(REPORTS/'monitor.html',page+'<p><a href="results.md">Results</a></p></html>')


def smoke(repo,deadline):
    environment=os.environ.copy();environment['PIDG_OBJECT_GPU_TEST']='1'
    environment.update(PIDG_HASH_SOURCE_DIR=str(engine.SOURCE/'hashencoder'),PIDG_HASH_BUILD_DIR=str(engine.SOURCE/'tmp_build'))
    check=subprocess.run([sys.executable,'-m','pytest','tests/test_object_id_gpu.py','-q'],cwd=ROOT,env=environment,capture_output=True,text=True)
    engine.write_text_atomic(REPORTS/'gpu_tests.log',check.stdout+check.stderr)
    if check.returncode:raise ValueError('Object CUDA parity/gradient tests failed')
    source=initialize('shared_prefix',True)
    if coordinator.step(source)<20:
        if base.train(source,repo,deadline,True,limit=20)!=75:raise ValueError('Smoke prefix did not stop at20')
    arms=[]
    for name in NAMES:
        d=initialize(name,True);branch(source,d,20);arms.append(d)
        if name==NAMES[-1] and coordinator.step(d)<70:
            if base.train(d,repo,deadline,True,limit=70)!=75:raise ValueError('Object resume boundary failed')
        if coordinator.step(d)<100 and base.train(d,repo,deadline,True)!=0:raise ValueError('Object smoke failed: '+name)
    if (arms[0]/'object_loss.jsonl').exists():raise ValueError('C1 control unexpectedly used object loss')
    rows=[json.loads(s) for s in (arms[-1]/'object_loss.jsonl').read_text().splitlines()]
    if not any(r.get('RGB_to_identity_gradient_norm',0)>0 for r in rows):raise ValueError('RGB reconstruction did not reach identity')
    if 'FULL_CHECKPOINT_RESTORED completed=70 next=71' not in (arms[-1]/'train.log').read_text():raise ValueError('Object optimizer resume not exercised')
    for d in arms:base.render(d,repo,deadline,split='validation',tags=['best'],smoke=True)
    from .identity_exports import export as identity_export
    identity_export(smoke=True)
    engine.write_json(REPORTS/'preflight.json',{'status':'passed','smoke_each':100,
        'resume_from':70,'RGB_reconstruction_reaches_identity':True,'validation_PNG_parity':True,
        'identity_export_validation_frames_each':30,
        'source_freeze_sha256':engine.sha256(REPORTS/'source_freeze.json'),'completed_utc':engine.now()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--hours',type=float,default=24.)
    args=parser.parse_args();REPORTS.mkdir(parents=True,exist_ok=True);RUNS.mkdir(parents=True,exist_ok=True)
    parent.assert_frozen(parent.HERE/'code');repo=prepare()
    if (REPORTS/'source_freeze.json').exists():assert_frozen(repo)
    else:engine.write_json(REPORTS/'source_freeze.json',freeze(repo))
    wire(repo)
    engine.write_json(REPORTS/'plan.json',{'names':NAMES,'configs':{n:configuration(n)[0] for n in NAMES},
        'same_prefix_full_checkpoint_sha256':engine.read_json(prefix()/'checkpoints/resume.json')['sha256'],
        'total_steps':50000,'split':[120,30,50],'no_instance_supervision':True,
        'C1_alpha_retained':True,'slot_capacity':16,'not_exact_paper_reproduction':True,'hours':args.hours})
    deadline=datetime.now(timezone.utc)+timedelta(hours=args.hours)
    state={'pid':os.getpid(),'status':'preflight','started_utc':engine.now(),'deadline_utc':deadline.isoformat(),'completed':[]}
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
            if proof.get('source_freeze_sha256')!=engine.sha256(REPORTS/'source_freeze.json'):smoke(repo,deadline)
            if args.smoke:state['status']='preflight_passed'
            else:
                state['status']='training';engine.write_json(RUNS/'controller.json',state)
                arms=[]
                for name in NAMES:
                    d=initialize(name);branch(prefix(),d,3000);arms.append(d)
                    if coordinator.step(d)<50000:
                        code=base.train(d,repo,deadline)
                        if code==75:raise SystemExit(75)
                        if code:raise ValueError('Training failed: '+name)
                    state['completed'].append(name);engine.write_json(RUNS/'controller.json',state)
                engine.write_json(RUNS/'selection_lock.json',{'locked_utc':engine.now(),
                    'validation_selections':{d.parent.name:engine.read_json(d/'metric_best.json') for d in arms},'tags':base.TAGS})
                state['status']='final_test';engine.write_json(RUNS/'controller.json',state)
                for d in arms:base.render(d,repo,deadline);base.pairs(d);coverage.coverage_scores(d)
                from variants.object_id.export import export
                export()
                from variants.object_id.identity_exports import export as identity_export
                identity_export()
                from variants.object_id.assess import assess
                assess()
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
