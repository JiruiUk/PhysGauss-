"""Observe live validation, propose a new recipe, test it, and promote only on evidence."""
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime,timezone
import psutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.integrity_tuning import run as parent
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from variants.centroid_tuning import run as coordinator
from .policy import propose,retry,review_final
from .prepare import prepare
HERE=Path(__file__).parent;RUNS=ROOT/'runs/pidg_integrity_feedback';REPORTS=ROOT/'experiments/integrity_feedback'
NAMES=['F1_feedback','F2_conservative_retry'];SMOKE='F0_smoke'
BASELINE=parent.REFERENCE


def configuration(name,smoke=False,off=False):
    config,text=parent.configuration(parent.NAMES[2],smoke,False)
    knobs={'integrity_tail_weight':.01,'integrity_tail_start':20,'integrity_tail_interval':1,'integrity_tail_fraction':.2,'integrity_tail_alpha_min':.05} if smoke else engine.read_json(REPORTS/'proposals.json')[name]['knobs']
    config.update(knobs);config['integrity_tail_audit']=smoke
    if smoke:config['coverage_density']=False
    if off:config.update(integrity_enabled=False,integrity_tail_weight=0.)
    return config,text


def freeze(repo):
    files=[*HERE.glob('*.py'),HERE/'launch.ps1',ROOT/'tests/test_integrity_feedback.py',ROOT/'tests/test_integrity_feedback_gpu.py',
        parent.REPORTS/'budget.json',parent.REPORTS/'diagnosis.json',ROOT/'experiments/training_monitor/index.html']
    return {'parent':parent.freeze(parent.HERE/'code'),'code':engine.runtime_hashes(repo),
        'helpers':{str(p.relative_to(ROOT)):engine.sha256(p) for p in files}}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json')!=freeze(repo):raise ValueError('Feedback source changed')


def wire(repo):
    engine.RUNS,engine.REPORTS,engine.repo_global,engine.assert_frozen=RUNS,REPORTS,repo,assert_frozen
    base.RUNS,base.REPORTS,base.assert_frozen=RUNS,REPORTS,assert_frozen
    coordinator.RUNS,coordinator.REPORTS,coordinator.configuration=RUNS,REPORTS,configuration
    coordinator.BASE='C1_repaired'


def branch(source,destination,boundary):
    receipt=destination/'branch.json'
    if receipt.exists():
        saved=engine.read_json(receipt)
        if saved['source']!=str(source) or saved['destination']!=str(destination) or saved['boundary']!=boundary:
            raise ValueError('Feedback branch provenance differs')
        return
    if coordinator.step(source)!=boundary:raise ValueError('Wrong feedback prefix boundary')
    if (source/'model_config.py').read_text()!=(destination/'model_config.py').read_text():raise ValueError('Changed original C1 recipe')
    a,b=[engine.read_json(d/'metric_config.json') for d in (source,destination)]
    changed={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    allowed={'coverage_density','coverage_add_limit','coverage_min_views','coverage_sample_interval'}
    if any(not k.startswith('integrity_') and k not in allowed for k in changed):raise ValueError('Unexpected feedback flags: '+str(changed))
    for folder in ('point_cloud','deform','checkpoints'):shutil.copytree(source/folder,destination/folder,dirs_exist_ok=True)
    for name in ('metric_best.json','metric_history.jsonl'):shutil.copyfile(source/name,destination/name)
    p=destination/'checkpoints/resume.pt';manifest=engine.read_json(p.with_suffix('.json'))
    if engine.sha256(p)!=manifest['sha256']:raise ValueError('Feedback full-state copy differs')
    state=engine.read_json(p.with_suffix('.metrics.json'))
    if state['config']!=a:raise ValueError('Wrong source selection/config state')
    state['config']=b;engine.write_json(p.with_suffix('.metrics.json'),state)
    engine.write_json(receipt,{'source':str(source),'destination':str(destination),'boundary':boundary,
        'identical_full_checkpoint_sha256':manifest['sha256'],'changed_flags':sorted(changed)})


def alive(pid,started=None):
    try:
        process=psutil.Process(int(pid))
        return process.is_running() and (not started or abs(process.create_time()-datetime.fromisoformat(started).timestamp())<30)
    except (psutil.Error,ValueError,TypeError):return False


def report(state):
    upstream=engine.read_json(parent.RUNS/'controller.json') if (parent.RUNS/'controller.json').exists() else {}
    active=engine.read_json(parent.RUNS/'active.json') if (parent.RUNS/'active.json').exists() else {}
    own=engine.read_json(RUNS/'active.json') if (RUNS/'active.json').exists() else {}
    progress={'iteration':0,'target':100 if '_smoke' in own.get('run','') else 50000}
    if own.get('run'):
        directory=Path(own['run']);progress['iteration']=base.progress(directory)
        match=re.findall(r'Training progress[^\r\n]*?\|\s*(\d+)/(\d+)',engine.tail(directory/'train.log',100000))
        if match:progress.update(iteration=max(progress['iteration'],int(match[-1][0])),target=int(match[-1][1]))
    result={'updated_utc':engine.now(),'controller':state.copy(),'upstream':upstream,'upstream_active':active,
        'upstream_controller_alive':alive(upstream.get('pid'),upstream.get('started_utc')),
        'upstream_child_alive':alive(active.get('pid'),active.get('started_utc')),'active':own,'active_progress':progress,
        'deadline_utc':engine.read_json(parent.REPORTS/'budget.json')['deadline_utc'],'reviews':[]}
    if active.get('run') and '_smoke' not in active['run']:
        history=parent.read_history(Path(active['run']))
        if history:
            row=history[-1];ref=next((r['metrics'] for r in base.history(BASELINE) if r['iteration']==row['iteration']),None)
            review={'model':active['id'],'iteration':row['iteration'],'metrics':row['metrics'],'integrity':row['integrity'],
                'reference_at_same_iteration':ref,'PSNR_delta_at_matched_iteration':row['metrics']['PSNR']-ref['PSNR'] if ref else None,
                'quality_guard':parent.quality_guard(row['metrics'],ref) if ref else None,'split':'validation'}
            result['reviews'].append(review)
            key=review['model']+'/'+str(review['iteration'])
            if key not in state.setdefault('seen_review_keys',[]):
                with (REPORTS/'observations.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps({'observed_utc':engine.now(),**review})+'\n')
                state['seen_review_keys'].append(key)
    rows=[]
    for name in NAMES:
        d=RUNS/name/'rubber_duck';h=parent.read_history(d)
        rows.append({'id':name,'iteration':base.progress(d),'history':h,'validation':h[-1] if h else None})
        if h:
            latest=h[-1];ref=next((r['metrics'] for r in base.history(BASELINE) if r['iteration']==latest['iteration']),None)
            own_review={'model':name,'iteration':latest['iteration'],'metrics':latest['metrics'],'integrity':latest['integrity'],
                'reference_at_same_iteration':ref,'PSNR_delta_at_matched_iteration':latest['metrics']['PSNR']-ref['PSNR'] if ref else None,
                'quality_guard':parent.quality_guard(latest['metrics'],ref) if ref else None,'split':'validation'}
            result['reviews'].insert(0,own_review);key=name+'/'+str(latest['iteration'])
            if key not in state.setdefault('seen_review_keys',[]):
                with (REPORTS/'observations.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps({'observed_utc':engine.now(),**own_review})+'\n')
                state['seen_review_keys'].append(key)
    result['rows']=rows
    engine.write_json(REPORTS/'results.json',result)
    lines=['# Validation review and model revision','',
        'State: '+state['status']+' | Deadline UTC: '+result['deadline_utc'],
        '','C1 baseline reused. Parent screens are observed every5 seconds. No running recipe is edited in place.',
        'Feedback proposals use validation only; changes create separate full-state3k branches.',
        'One measured revision and at most one conservative retry; at most one additional50k promotion.','']
    for r in result['reviews']:
        lines += [f"Latest: {r['model']} / {r['iteration']} iter | PSNR delta vs C1 at same iter: {r['PSNR_delta_at_matched_iteration']:+.5f} dB" if r['PSNR_delta_at_matched_iteration'] is not None else f"Latest: {r['model']}",
            'Quality guard: '+str(r['quality_guard'])+' | Integrity score: '+str(r['integrity']['integrity_score'])]
    if (REPORTS/'proposals.json').exists():
        for name,p in engine.read_json(REPORTS/'proposals.json').items():lines += ['',name+': '+p['action'],*['- '+r for r in p['reasons']]]
    if state.get('error'):lines+=['','Error: '+state['error']]
    engine.write_text_atomic(REPORTS/'results.md','\n'.join(lines)+'\n')
    engine.write_text_atomic(REPORTS/'monitor.html','<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>Validation feedback</title><style>body{background:#142033;color:#edf5ff;font:16px system-ui;margin:24px}pre{white-space:pre-wrap}a{color:#8de4b1}</style><h1>验证复核与模型修改</h1><p>Updated UTC: '+result['updated_utc']+f"</p><h2>{own.get('id','waiting')} | {progress['iteration']:,} / {progress['target']:,} iter</h2><pre>"+__import__('html').escape('\n'.join(lines))+'</pre>'+''.join(f"<p>{r['id']}: {r['iteration']:,}/50,000 iter</p>" for r in rows)+'<a href="../training_monitor/index.html">统一监控</a>')


def ensure_time(deadline):
    if datetime.now(timezone.utc)>=deadline or (RUNS/'stop_queue.request').exists():raise SystemExit(75)


def preflight(repo,deadline):
    ensure_time(deadline)
    env=os.environ.copy();env.update(PIDG_FEEDBACK_GPU_TEST='1',PIDG_HASH_SOURCE_DIR=str(engine.SOURCE/'hashencoder'),PIDG_HASH_BUILD_DIR=str(engine.SOURCE/'tmp_build'))
    check=subprocess.run([sys.executable,'-m','pytest','tests/test_integrity_feedback_gpu.py','-q'],cwd=ROOT,env=env,capture_output=True,text=True)
    engine.write_text_atomic(REPORTS/'gpu_tests.log',check.stdout+check.stderr)
    if check.returncode:raise ValueError('Feedback color-gradient CUDA check failed')
    d=coordinator.initialize(SMOKE,True);branch(parent.RUNS/'_smoke/shared_prefix/rubber_duck',d,20)
    if coordinator.step(d)<70 and base.train(d,repo,deadline,True,limit=70)!=75:raise ValueError('Feedback short pause failed')
    ensure_time(deadline)
    if coordinator.step(d)<100 and base.train(d,repo,deadline,True)!=0:raise ValueError('Feedback short training failed')
    if b'FULL_CHECKPOINT_RESTORED completed=70 next=71' not in (d/'train.log').read_bytes():raise ValueError('Feedback resume not exercised')
    events=[json.loads(line) for line in (d/'foreground_tail.jsonl').read_text(encoding='utf-8').splitlines()]
    if not any(e.get('SH_gradient_norm',0)>0 for e in events):raise ValueError('Tail objective did not reach actual Gaussian color')
    base.render(d,repo,deadline,split='validation',tags=['best'],smoke=True)
    engine.write_json(REPORTS/'preflight.json',{'status':'passed','short_steps':100,'resume_from':70,
        'actual_color_gradient':True,'validation_png_parity':True,'source_freeze_sha256':engine.sha256(REPORTS/'source_freeze.json')})


def commit_preflight():
    paths=['variants/integrity_feedback','tests/test_integrity_feedback.py','tests/test_integrity_feedback_gpu.py',
        'experiments/training_monitor/index.html',*['experiments/integrity_feedback/'+n for n in ('source_freeze.json','unit_tests.json','preflight.json','gpu_tests.log','proposals.json')]]
    subprocess.run(['git','add','--',*paths],cwd=ROOT,capture_output=True,check=True)
    if subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=ROOT).returncode==0:return
    result=subprocess.run(['git','-c','core.whitespace=-blank-at-eol,-blank-at-eof','diff','--cached','--check','--',*paths],cwd=ROOT,capture_output=True,text=True)
    if result.returncode:raise ValueError('Feedback staged whitespace errors')
    result=subprocess.run(['git','commit','--only','-m','Add validated result-driven integrity feedback experiment','--',*paths],cwd=ROOT,capture_output=True,text=True)
    engine.write_json(REPORTS/'milestone_commit.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    if result.returncode:raise ValueError('Feedback milestone commit failed')


def train_screen(name,repo,deadline):
    ensure_time(deadline);d=coordinator.initialize(name)
    branch(parent.parent.RUNS/'shared_prefix/rubber_duck',d,3000)
    if coordinator.step(d)<20000:
        code=base.train(d,repo,deadline,limit=20000);ensure_time(deadline)
        if code!=75 or coordinator.step(d)!=20000:raise ValueError('Feedback screen did not reach20k')
    return parent.render_screen(d,repo,deadline)


def main():
    REPORTS.mkdir(parents=True,exist_ok=True);RUNS.mkdir(parents=True,exist_ok=True)
    parent.assert_frozen(parent.HERE/'code');repo=prepare()
    deadline=datetime.fromisoformat(engine.read_json(parent.REPORTS/'budget.json')['deadline_utc'])
    if not (REPORTS/'source_freeze.json').exists():engine.write_json(REPORTS/'source_freeze.json',freeze(repo))
    else:assert_frozen(repo)
    wire(repo)
    state={'pid':os.getpid(),'status':'observing_parent_screens','started_utc':engine.now(),'deadline_utc':deadline.isoformat()}
    observations=REPORTS/'observations.jsonl'
    if observations.exists():
        state['seen_review_keys']=list({r['model']+'/'+str(r['iteration']) for r in (json.loads(line) for line in observations.read_text(encoding='utf-8').splitlines())})
    stop=threading.Event()
    def sample():
        while not stop.wait(5):
            try:report(state)
            except (OSError,ValueError,KeyError):pass
    with engine.lock():
        engine.write_json(RUNS/'controller.json',state);thread=threading.Thread(target=sample,daemon=True);thread.start();report(state)
        os.environ['PIDG_EXPORT_ALPHA']='1'
        try:
            while not (parent.REPORTS/'selection.json').exists():
                ensure_time(deadline)
                upstream=engine.read_json(parent.RUNS/'controller.json')
                if upstream['status']=='failed':raise RuntimeError('Upstream failed; inspect integrity monitor before changing the model: '+upstream.get('error',''))
                time.sleep(5)
            selection=engine.read_json(parent.REPORTS/'selection.json')
            if selection.get('split')!='validation' or any(r['iteration']!=20000 for r in selection['screens']):
                raise ValueError('Feedback decisions require matched20k validation screens')
            configs={r['model']:engine.read_json(parent.RUNS/r['model']/'rubber_duck/metric_config.json') for r in selection['screens']}
            reference=next(r['metrics'] for r in base.history(BASELINE) if r['iteration']==20000)
            geometry=engine.read_json(parent.REPORTS/'diagnosis.json')['mean']
            proposal=propose(selection['screens'],configs,reference,geometry)
            engine.write_json(REPORTS/'proposals.json',{NAMES[0]:proposal})
            engine.write_json(REPORTS/'proposal_provenance.json',{'source_split':'validation','parent_selection_sha256':engine.sha256(parent.REPORTS/'selection.json'),'parent_selection_file':str(parent.REPORTS/'selection.json')})
            state['status']='proposal_ready_waiting_for_GPU';engine.write_json(RUNS/'controller.json',state)
            if proposal['action']!='train':state['status']='completed_no_supported_change';return
            while True:
                ensure_time(deadline);upstream=engine.read_json(parent.RUNS/'controller.json')
                if upstream['status'] in ('completed','completed_no_safe_promotion'):break
                if upstream['status']=='failed':raise RuntimeError('Upstream failed before releasing GPU: '+upstream.get('error',''))
                time.sleep(5)
            completed=next((r for r in reversed(parent.read_history(parent.RUNS/proposal['source_model']/'rubber_duck')) if r['iteration']==50000),None)
            proposal=review_final(proposal,completed,engine.read_json(parent.REPORTS/'diagnosis.json')['image_metrics'],geometry)
            engine.write_json(REPORTS/'proposals.json',{NAMES[0]:proposal})
            if (deadline-datetime.now(timezone.utc)).total_seconds()<3600:
                state['status']='insufficient_budget_for_safe_followup';return
            state['status']='feedback_preflight';engine.write_json(RUNS/'controller.json',state)
            proof=engine.read_json(REPORTS/'preflight.json') if (REPORTS/'preflight.json').exists() else {}
            if proof.get('source_freeze_sha256')!=engine.sha256(REPORTS/'source_freeze.json'):preflight(repo,deadline)
            commit_preflight();outcomes=[];promoted=None
            for index,name in enumerate(NAMES):
                state.update(status='feedback_screen',candidate=name);engine.write_json(RUNS/'controller.json',state)
                outcome=train_screen(name,repo,deadline);outcomes.append(outcome)
                passed=parent.quality_guard(outcome['metrics'],reference) and outcome['integrity']['background_false_coverage']<=geometry['background_false_coverage']+.005
                improved=outcome['integrity']['integrity_score']<proposal['source_geometry']['integrity_score']-1e-5
                engine.write_json(REPORTS/'screen_reviews.json',{'split':'validation','outcomes':outcomes,'last_quality_guard':passed,'last_improved_at_matched_20k':improved})
                if passed and improved:promoted=name;break
                if index or (deadline-datetime.now(timezone.utc)).total_seconds()<5400:break
                revision=retry(proposal,outcome,reference,geometry)
                if revision['action']!='train':break
                proposals=engine.read_json(REPORTS/'proposals.json');proposals[NAMES[1]]=revision;engine.write_json(REPORTS/'proposals.json',proposals)
            if promoted is None:state['status']='completed_no_validated_improvement';return
            state.update(status='feedback_long_training',promoted=promoted);engine.write_json(RUNS/'controller.json',state)
            d=RUNS/promoted/'rubber_duck'
            if coordinator.step(d)<50000:
                code=base.train(d,repo,deadline)
                if code==75:raise SystemExit(75)
                if code:raise ValueError('Feedback long training failed')
            engine.write_json(RUNS/'selection_lock.json',{'locked_utc':engine.now(),'model':promoted,'validation_selection':engine.read_json(d/'metric_best.json'),
                'integrity_selection':engine.read_json(d/'integrity_best.json') if (d/'integrity_best.json').exists() else None})
            state['status']='feedback_final_test';engine.write_json(RUNS/'controller.json',state);ensure_time(deadline)
            tags=['50000','best']+(['integrity_best'] if (d/'integrity_best.json').exists() else [])
            base.render(d,repo,deadline,tags=tags)
            from .export import export_and_review
            export_and_review(promoted,engine.read_json(REPORTS/'proposals.json')[promoted]);state['status']='completed'
            assert_frozen(repo)
        except SystemExit as exc:
            state['status']='saved_stopped' if exc.code==75 else 'failed'
            if exc.code!=75:raise
        except Exception as exc:
            state.update(status='failed',error=str(exc));raise
        finally:
            state['finished_utc']=engine.now();engine.write_json(RUNS/'controller.json',state);stop.set();thread.join(timeout=6);report(state)


if __name__=='__main__':main()
