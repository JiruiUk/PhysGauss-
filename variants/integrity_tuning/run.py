"""Eight-hour queue: diagnose, wait for current job, preflight, screen, promote, export."""
import argparse
from datetime import datetime,timezone
import html
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.color_repair import run as parent
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from variants.centroid_tuning import run as coordinator
from .prepare import prepare
from .diagnose import diagnose,evaluate_exports,PREVIEW
from .runtime import quality_guard
HERE=Path(__file__).parent;RUNS=ROOT/'runs/pidg_integrity_tuning';REPORTS=ROOT/'experiments/integrity_tuning'
NAMES=['H1_balanced_coverage','H2_temporal_cohesion','H3_combined','H4_color_density']
RETRY='H5_weaker_combined'
REFERENCE=parent.RUNS/'C1_repaired/rubber_duck'
UPSTREAM=ROOT/'runs/pidg_object_id'


def configuration(name,smoke=False,off=False):
    c,text=parent.configuration('C1_repaired',smoke,False)
    active=name in [*NAMES,RETRY] and not off
    c.update(integrity_enabled=active,integrity_coverage_weight=(.0005 if name==RETRY else .002) if active and name in (NAMES[0],NAMES[2],RETRY) else 0.,
        integrity_strain_weight=(.000025 if name==RETRY else .0001) if active and name in (NAMES[1],NAMES[2],RETRY) else 0.,
        integrity_coverage_start=20 if smoke else 3000,integrity_strain_start=30 if smoke else 8000,
        integrity_ramp=10 if smoke else 4000,integrity_strain_interval=5 if smoke else 20,
        integrity_screen_iteration=100 if smoke else 20000,
        integrity_baseline_metrics=engine.read_json(REPORTS/'diagnosis.json')['image_metrics'],
        integrity_baseline_diagnostics=engine.read_json(REPORTS/'diagnosis.json')['mean'],
        coverage_density=(name in (NAMES[3],'shared_prefix')) and not off)
    if smoke:c.update(coverage_min_views=1,coverage_sample_interval=1)
    return c,text


def freeze(repo):
    files=[*HERE.glob('*.py'),HERE/'launch.ps1',ROOT/'tests/test_integrity_tuning.py',ROOT/'tests/test_integrity_gpu.py',REPORTS/'budget.json',REPORTS/'diagnosis.json']
    files.append(ROOT/'scripts/serve_integrity_monitor.py')
    files += [PREVIEW/p for p in ('manifest.json','input_manifest.json','results.json','per_view.json')]
    files += [REFERENCE/p for p in ('model_config.py','metric_config.json','metric_history.jsonl','checkpoints/resume.json','branch.json')]
    files += [parent.RUNS/'shared_prefix/rubber_duck'/p for p in ('model_config.py','metric_config.json','checkpoints/resume.json','checkpoints/resume.metrics.json')]
    return {'parent':parent.freeze(parent.HERE/'code'),'code':engine.runtime_hashes(repo),
        'helpers_inputs':{str(p.relative_to(ROOT)):engine.sha256(p) for p in files}}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json')!=freeze(repo):raise ValueError('Integrity source/data changed')


def wire(repo):
    base.RUNS,base.REPORTS,base.assert_frozen=RUNS,REPORTS,assert_frozen
    engine.RUNS,engine.REPORTS,engine.repo_global,engine.assert_frozen=RUNS,REPORTS,repo,assert_frozen
    coordinator.RUNS,coordinator.REPORTS,coordinator.configuration=RUNS,REPORTS,configuration
    coordinator.BASE='C1_repaired'


def initialize(name,smoke=False):return coordinator.initialize(name,smoke)


def branch(source,destination,boundary):
    if (destination/'branch.json').exists():
        receipt=engine.read_json(destination/'branch.json')
        if receipt['source']!=str(source) or receipt['boundary']!=boundary or \
            receipt['identical_full_checkpoint_sha256']!=engine.read_json(source/'checkpoints/resume.json')['sha256']:
            raise ValueError('Existing integrity branch provenance differs')
        if receipt.get('destination',str(destination))!=str(destination):raise ValueError('Wrong branch destination')
        if 'destination' not in receipt:
            receipt['destination']=str(destination);engine.write_json(destination/'branch.json',receipt)
        return
    if (destination/'checkpoints/resume.pt').exists():raise ValueError('Unreceipted integrity branch')
    if coordinator.step(source)!=boundary:raise ValueError('Wrong integrity branch boundary')
    if (source/'model_config.py').read_text()!=(destination/'model_config.py').read_text():raise ValueError('C1 optimization recipe changed')
    a,b=[engine.read_json(d/'metric_config.json') for d in (source,destination)]
    changed={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    if any(not k.startswith('integrity_') and k not in ('coverage_density','coverage_min_views','coverage_sample_interval') for k in changed):
        raise ValueError('Unexpected integrity branch flags: '+str(changed))
    for folder in ('point_cloud','deform','checkpoints'):shutil.copytree(source/folder,destination/folder,dirs_exist_ok=True)
    for name in ('metric_best.json','metric_history.jsonl'):shutil.copyfile(source/name,destination/name)
    checkpoint=destination/'checkpoints/resume.pt';manifest=engine.read_json(checkpoint.with_suffix('.json'))
    if engine.sha256(checkpoint)!=manifest['sha256']:raise ValueError('Changed shared full checkpoint')
    state=engine.read_json(checkpoint.with_suffix('.metrics.json'))
    if state['config']!=a:raise ValueError('Wrong prefix config')
    state['config']=b;engine.write_json(checkpoint.with_suffix('.metrics.json'),state)
    engine.write_json(destination/'branch.json',{'source':str(source),'destination':str(destination),'boundary':boundary,
        'identical_full_checkpoint_sha256':manifest['sha256'],'changed_flags':sorted(changed)})


def read_history(directory):
    path=directory/'integrity_history.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def report():
    state=engine.read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {}
    active=engine.read_json(RUNS/'active.json') if (RUNS/'active.json').exists() else {}
    active_progress={'iteration':0,'target':100 if '_smoke' in active.get('run','') else 50000}
    failure_detail=''
    if active.get('run'):
        directory=Path(active['run']);log=engine.tail(directory/'train.log',100000)
        matches=re.findall(r'Training progress[^\r\n]*?\|\s*(\d+)/(\d+)',log)
        if matches:active_progress.update(iteration=int(matches[-1][0]),target=int(matches[-1][1]))
        checkpoint=directory/'checkpoints/resume.json'
        if checkpoint.exists():active_progress['iteration']=max(active_progress['iteration'],engine.read_json(checkpoint)['iteration'])
        errors=re.findall(r'^([\w.]+(?:Error|Exception):[^\r\n]*)',log.rsplit('\nCOMMAND ',1)[-1],re.MULTILINE)
        if state.get('status')=='failed' and errors:failure_detail=errors[-1]
    rows=[]
    for name in [*NAMES,RETRY]:
        d=RUNS/name/'rubber_duck';events=read_history(d)
        iteration=base.progress(d)
        status='completed' if iteration>=50000 else 'queued_screen'
        if name==RETRY and not iteration:status='optional_retry'
        if name in state.get('completed_screens',[]):status='screen_complete'
        if name in state.get('promoted',[]) and iteration<50000:status='queued_long'
        if name==active.get('id'):status='running' if state.get('status') not in ('completed','saved_stopped','failed') else status
        if name in state.get('failed_screens',{}):status='rejected_failed'
        if iteration>=50000:status='completed'
        rows.append({'id':name,'iteration':iteration,'status':status,'history':events,
            'validation':events[-1] if events else None,'source':str(d)})
    result={'updated_utc':engine.now(),'controller':state,'active':active,'active_progress':active_progress,
        'failure_detail':failure_detail,'rows':rows,
        'baseline_reused_from':str(REFERENCE),'priority':'completeness with appearance safeguards',
        'budget':engine.read_json(REPORTS/'budget.json'),'split':[120,30,50]}
    engine.write_json(REPORTS/'results.json',result)
    lines=['# Completeness-first C1 tuning','',
        'Status: '+state.get('status','preparing')+' | Active: '+active.get('id','-')+' / '+active.get('phase','-'),
        f"Active iteration: {active_progress['iteration']} / {active_progress['target']} | Updated UTC: {result['updated_utc']}",
        '','C1 baseline weights, metrics and renders reused. No baseline retraining.',
        'Eight-hour deadline includes waiting for the current Object-ID queue; GPU work is sequential.',
        'Four20k screens, at most two50k promotions.120 train /30 validation /50 final test.',
        'Only training RGB/alpha supervise losses. No instance IDs or learned material segmentation.',
        'Completeness metrics are2D proxies; they do not prove watertight3D geometry.','',
        '| Model | State | Iter | Validation PSNR | Integrity score (lower) | Core low90 | BG spill |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        v=r['validation'] or {};m=v.get('metrics',{});g=v.get('integrity',{})
        lines.append(f"| {r['id']} | {r['status']} | {r['iteration']}/50000 | {m.get('PSNR','-')} | {g.get('integrity_score','-')} | {g.get('core_low_90','-')} | {g.get('background_false_coverage','-')} |")
    if state.get('error'):lines+=['','Queue error: '+state['error'],'Detail: '+failure_detail]
    lines+=['','[Diagnosis](diagnosis.md) | [Monitor](monitor.html) | [Plan](plan.json)',
        '','Deadline UTC: '+result['budget']['deadline_utc']]
    engine.write_text_atomic(REPORTS/'results.md','\n'.join(lines)+'\n')
    page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>Object completeness</title><style>body{font:16px system-ui;background:#142033;color:#edf5ff;margin:24px}section,svg{background:#203047;padding:12px;margin:12px 0}progress{width:85%}a{color:#8de4b1}</style><h1>Object completeness / C1</h1><p>'+html.escape(state.get('status','preparing')+' | '+active.get('id','-')+' / '+active.get('phase','-'))+'</p><p>Deadline UTC: '+result['budget']['deadline_utc']+' | C1 reused; no baseline training.</p>'
    page+=f"<section><h2>当前任务：{html.escape(active.get('id','-'))}</h2><h2>{active_progress['iteration']:,} / {active_progress['target']:,} iter</h2><p>{'预检短训练，正式筛选尚未开始' if '_smoke' in active.get('run','') else '正式实验进度'}</p><p>每5秒刷新 | 更新时间 UTC：{result['updated_utc']} | 数据距今 <span id='age'></span>秒 | 剩余预算 <span id='remaining'></span></p></section>"
    if state.get('error'):
        page+='<section style="border:2px solid #ff7979"><h2>队列已停止：错误</h2><pre style="white-space:pre-wrap">'+html.escape(state['error']+'\n'+failure_detail)+'</pre><p>详细状态：<a href="results.json">results.json</a></p></section>'
    for r in rows:
        page+=f"<section><b>{r['id']} | {r['status']}</b><h2>{r['iteration']:,} /50,000 iter</h2><progress value='{r['iteration']}' max='50000'></progress></section>"
    chart_rows=rows
    short=not any(r['history'] for r in rows) and bool(active.get('run')) and '_smoke' in active['run']
    if short:
        chart_rows=[{'id':'preflight '+active['id'],'history':read_history(Path(active['run']))}]
        page+='<h2>预检验证曲线（100 iter，不代表正式模型性能）</h2>'
    xmax=100 if short else 50000
    for label,field in [('Validation PSNR','PSNR'),('Validation SSIM','SSIM'),('Validation LPIPS (lower)','LPIPS'),('Validation MS-SSIM','MS-SSIM'),('Integrity score (lower)','integrity_score'),('Core low coverage50 (lower)','core_low_50'),('Core low coverage90 (lower)','core_low_90'),('Fragment loss (lower)','fragment_loss'),('Opaque appearance error (lower)','opaque_appearance_error'),('Background spill (lower)','background_false_coverage')]:
        series=[[{'iteration':h['iteration'],'value':h['metrics'][field] if field in engine.METRICS else h['integrity'][field]} for h in r['history']] for r in chart_rows]
        values=[h['value'] for s in series for h in s]
        if not values:continue
        lo,hi=min(values),max(values);span=max(1e-8,hi-lo)
        page+='<h2>'+label+'</h2><svg viewBox="0 0 1000 300" width="1000"><text x="5" y="20" fill="white">'+f'{hi:.6f}</text><text x="5" y="240" fill="white">{lo:.6f}</text>'
        for i,(s,color) in enumerate(zip(series,['#86d1ff','#8de4b1','#ffbe83','#e9a0ee','#ccc'])):
            coordinates=[(80+880*h['iteration']/xmax,240-200*(h['value']-lo)/span) for h in s]
            points=' '.join(f'{x:.1f},{y:.1f}' for x,y in coordinates)
            page+=f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'+''.join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="{color}"/>' for x,y in coordinates)+f'<text x="{60+i*190}" y="280" fill="{color}">{html.escape(chart_rows[i]["id"])}</text>'
        page+='</svg>'
    page+='<script>const updated='+json.dumps(result['updated_utc'])+',deadline='+json.dumps(result['budget']['deadline_utc'])+';function tick(){document.getElementById("age").textContent=Math.max(0,Math.floor((Date.now()-Date.parse(updated))/1000));let s=Math.max(0,Math.floor((Date.parse(deadline)-Date.now())/1000));document.getElementById("remaining").textContent=Math.floor(s/3600)+"h "+Math.floor(s%3600/60)+"m"}tick();setInterval(tick,1000);</script>'
    engine.write_text_atomic(REPORTS/'monitor.html',page+'<p><a href="diagnosis.md">Diagnosis</a> | <a href="results.md">Results</a> | <a href="../object_id/monitor.html">Object-ID results</a></p>')


def check_budget(deadline):
    if datetime.now(timezone.utc)>=deadline or (RUNS/'stop_queue.request').exists():raise SystemExit(75)


def wait_upstream(state,deadline):
    while True:
        check_budget(deadline)
        upstream=engine.read_json(UPSTREAM/'controller.json')
        state.update(status='waiting_for_object_id',upstream_status=upstream['status'])
        engine.write_json(RUNS/'controller.json',state)
        if upstream['status']=='completed':return
        if upstream['status'] in ('failed','saved_stopped'):
            raise RuntimeError('Upstream Object-ID queue not completed: '+json.dumps(upstream))
        time.sleep(5)


def preflight(repo,deadline):
    check_budget(deadline)
    env=os.environ.copy();env.update(PIDG_INTEGRITY_GPU_TEST='1',PIDG_COLOR_REPAIR_GPU_TEST='1',
        PIDG_HASH_SOURCE_DIR=str(engine.SOURCE/'hashencoder'),PIDG_HASH_BUILD_DIR=str(engine.SOURCE/'tmp_build'))
    check=subprocess.run([sys.executable,'-m','pytest','tests/test_integrity_gpu.py','tests/test_color_repair_gpu.py','-q'],cwd=ROOT,env=env,capture_output=True,text=True)
    engine.write_text_atomic(REPORTS/'gpu_tests.log',check.stdout+check.stderr)
    if check.returncode:raise ValueError('Integrity CUDA preflight failed')
    source=initialize('shared_prefix',True)
    if coordinator.step(source)<20:
        if base.train(source,repo,deadline,True,limit=20)!=75:raise ValueError('Short C1 prefix did not stop at20')
    for name in (NAMES[2],NAMES[3]):
        check_budget(deadline)
        d=initialize(name,True);branch(source,d,20)
        if coordinator.step(d)<70 and base.train(d,repo,deadline,True,limit=70)!=75:raise ValueError('Smoke pause failed')
        check_budget(deadline)
        if coordinator.step(d)<100 and base.train(d,repo,deadline,True)!=0:raise ValueError('Smoke training failed')
        if b'FULL_CHECKPOINT_RESTORED completed=70 next=71' not in (d/'train.log').read_bytes():raise ValueError('Resume was not exercised')
        if len(read_history(d)[-1]['per_frame'])!=30:raise ValueError('Incomplete short integrity evaluation')
        base.render(d,repo,deadline,split='validation',tags=['best'],smoke=True)
    density=RUNS/'_smoke'/NAMES[3]/'rubber_duck/coverage_density.jsonl'
    if not any(json.loads(line)['added']>0 for line in density.read_text().splitlines()):raise ValueError('Color-consistent density not exercised')
    events=[json.loads(line) for line in (RUNS/'_smoke'/NAMES[2]/'rubber_duck/integrity_loss.jsonl').read_text().splitlines()]
    if not any(r.get('valid_edges',0)>0 for r in events):raise ValueError('Temporal neighbor gate had no valid edges')
    manifest=engine.read_json(HERE/'source_manifest.json')
    different=[name for name in manifest['parent'] if manifest['parent'][name]!=manifest['fork'][name]]
    if different!=['train_pidg.py']:raise ValueError('Unexpected inference change')
    engine.write_json(REPORTS/'preflight.json',{'status':'passed','smoke_steps':100,'resume_from':70,
        'density_exercised':True,'cohesion_exercised':True,'validation_png_parity':True,
        'inference_runtime_identical_to_C1':True,'source_freeze_sha256':engine.sha256(REPORTS/'source_freeze.json'),'completed_utc':engine.now()})


def commit_preflight():
    paths=['variants/integrity_tuning','tests/test_integrity_tuning.py','tests/test_integrity_gpu.py','scripts/serve_integrity_monitor.py',
        *['experiments/integrity_tuning/'+name for name in ('budget.json','plan.json','diagnosis.json','diagnosis.md','unit_tests.json','gpu_tests.log','preflight.json','source_freeze.json')]]
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True)
    if subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=ROOT).returncode==0:
        engine.write_json(REPORTS/'milestone_commit.json',{'status':'already_committed'});return
    result=subprocess.run(['git','-c','core.whitespace=-blank-at-eol,-blank-at-eof','diff','--cached','--check','--',*paths],cwd=ROOT,capture_output=True,text=True)
    if result.returncode:raise ValueError('Integrity staged whitespace check failed: '+result.stdout)
    result=subprocess.run(['git','commit','--only','-m','Add validated completeness-first C1 ablation queue','--',*paths],cwd=ROOT,capture_output=True,text=True)
    engine.write_json(REPORTS/'milestone_commit.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    if result.returncode and 'nothing to commit' not in result.stdout:raise ValueError('Integrity milestone commit failed')


def render_screen(d,repo,deadline):
    tag='screen20000';raw=d/'test'/('ours_'+tag)
    if not (raw/'render_complete.json').exists():
        command=[*engine.command_prefix(repo),'render.py','-s',str(base.DATA/'training'),'-m',str(d),
            '--conf',str(d/'model_config.py'),'--load2gpu_on_the_fly','--resolution','1600','--iteration',tag,'--skip_train']
        engine.execute(command,d,'render_validation_'+tag,engine.env_for(d),deadline)
        engine.write_json(raw/'render_complete.json',{'split':'validation','images':30})
    engine.execute([*engine.command_prefix(repo),'metrics.py','-m',str(d)],d,'metrics_screen',engine.env_for(d),deadline)
    metrics=engine.read_json(d/'results.json');per=engine.read_json(d/'per_view.json');base.verify_metrics(metrics,per,[tag],30)
    geometry=evaluate_exports(d,tag)
    logged=next(r for r in reversed(read_history(d)) if r['iteration']==20000)
    if abs(geometry['mean']['integrity_score']-logged['integrity']['integrity_score'])>1e-6:raise ValueError('Integrity PNG/validation mismatch')
    record={'model':d.parent.name,'iteration':20000,'metrics':metrics['ours_'+tag],'integrity':geometry['mean'],'per_frame':geometry['per_frame']}
    engine.write_json(d/'screen.json',record);return record


def screen(name,repo,deadline):
    check_budget(deadline);d=initialize(name);branch(parent.RUNS/'shared_prefix/rubber_duck',d,3000)
    if coordinator.step(d)<20000:
        code=base.train(d,repo,deadline,limit=20000)
        check_budget(deadline)
        if code!=75 or coordinator.step(d)!=20000:raise ValueError('Screen failed to reach exactly20k')
    return render_screen(d,repo,deadline)


def select(screens):
    reference=next(r['metrics'] for r in base.history(REFERENCE) if r['iteration']==20000)
    candidates=[]
    baseline=engine.read_json(REPORTS/'diagnosis.json')['mean']
    for r in screens:
        r['quality_guard_at_matched_20k']=quality_guard(r['metrics'],reference) and \
            r['integrity']['background_false_coverage']<=baseline['background_false_coverage']+.005
        if r['quality_guard_at_matched_20k']:candidates.append(r)
    candidates.sort(key=lambda r:(r['integrity']['integrity_score'],r['integrity']['p95_score'],-r['metrics']['PSNR']))
    return [r['model'] for r in candidates[:2]]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    RUNS.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    budget=engine.read_json(REPORTS/'budget.json');deadline=datetime.fromisoformat(budget['deadline_utc'])
    parent.assert_frozen(parent.HERE/'code');repo=prepare()
    if engine.read_json(REFERENCE/'branch.json')['identical_full_checkpoint_sha256']!=engine.read_json(parent.RUNS/'shared_prefix/rubber_duck/checkpoints/resume.json')['sha256']:
        raise ValueError('C1 reference and new variants must share the identical3k state')
    if not (REPORTS/'diagnosis.json').exists():diagnose()
    if (REPORTS/'source_freeze.json').exists():assert_frozen(repo)
    else:engine.write_json(REPORTS/'source_freeze.json',freeze(repo))
    wire(repo)
    engine.write_json(REPORTS/'plan.json',{'candidates':{n:configuration(n)[0] for n in [*NAMES,RETRY]},
        'baseline_reused':str(REFERENCE),'prefix_sha256':engine.read_json(parent.RUNS/'shared_prefix/rubber_duck/checkpoints/resume.json')['sha256'],
        'screen_total':20000,'promote_total':50000,'maximum_promotions':2,'split':[120,30,50],
        'selection':'validation completeness with matched-iteration four-metric guards; test only after all selections locked','budget':budget})
    state={'pid':os.getpid(),'status':'preparing','started_utc':engine.now(),'deadline_utc':budget['deadline_utc'],'completed_screens':[],'completed_long':[]}
    event=threading.Event()
    def monitor():
        while not event.wait(5):
            try:report()
            except (OSError,ValueError,KeyError):pass
    with engine.lock():
        engine.write_json(RUNS/'controller.json',state);thread=threading.Thread(target=monitor,daemon=True);thread.start();report()
        os.environ['PIDG_EXPORT_ALPHA']='1'
        try:
            wait_upstream(state,deadline)
            state['status']='preflight';engine.write_json(RUNS/'controller.json',state)
            proof=engine.read_json(REPORTS/'preflight.json') if (REPORTS/'preflight.json').exists() else {}
            if proof.get('source_freeze_sha256')!=engine.sha256(REPORTS/'source_freeze.json'):preflight(repo,deadline)
            commit_preflight()
            if args.preflight_only:state['status']='preflight_passed';return
            screens=[]
            for name in NAMES:
                state.update(status='screening',candidate=name);engine.write_json(RUNS/'controller.json',state)
                try:
                    screens.append(screen(name,repo,deadline));state['completed_screens'].append(name)
                except RuntimeError as exc:
                    check_budget(deadline)
                    state.setdefault('failed_screens',{})[name]=str(exc)
                    engine.write_json(REPORTS/('failure_'+name+'.json'),{'model':name,'error':str(exc),
                        'training_log':str(RUNS/name/'rubber_duck/train.log'),'action':'reject this candidate; continue other methods'})
            promoted=select(screens)
            if not promoted and (deadline-datetime.now(timezone.utc)).total_seconds()>7200:
                screens.append(screen(RETRY,repo,deadline));state['completed_screens'].append(RETRY);promoted=select(screens)
            engine.write_json(REPORTS/'selection.json',{'split':'validation','screens':screens,'promoted':promoted,'selected_utc':engine.now()})
            if not promoted:state['status']='completed_no_safe_promotion';return
            state.update(status='long_training',promoted=promoted);engine.write_json(RUNS/'controller.json',state)
            for name in promoted:
                check_budget(deadline);d=RUNS/name/'rubber_duck'
                if coordinator.step(d)<50000:
                    code=base.train(d,repo,deadline)
                    if code==75:raise SystemExit(75)
                    if code:raise ValueError('Integrity long training failed')
                state['completed_long'].append(name);engine.write_json(RUNS/'controller.json',state)
            engine.write_json(RUNS/'selection_lock.json',{'locked_utc':engine.now(),'promoted':promoted,
                'validation_selections':{n:engine.read_json(RUNS/n/'rubber_duck/metric_best.json') for n in promoted},
                'integrity_selections':{n:engine.read_json(RUNS/n/'rubber_duck/integrity_best.json') if (RUNS/n/'rubber_duck/integrity_best.json').exists() else None for n in promoted}})
            state['status']='final_test';engine.write_json(RUNS/'controller.json',state)
            for name in promoted:
                check_budget(deadline);d=RUNS/name/'rubber_duck'
                tags=['50000','best']+(['integrity_best'] if (d/'integrity_best.json').exists() else [])
                base.render(d,repo,deadline,tags=tags)
            from .export import export_and_assess
            export_and_assess(promoted);state['status']='completed'
            assert_frozen(repo)
        except SystemExit as exc:
            state['status']='saved_stopped' if exc.code==75 else 'failed'
            if exc.code!=75:raise
        except Exception as exc:
            state.update(status='failed',error=str(exc));raise
        finally:
            state['finished_utc']=engine.now();engine.write_json(RUNS/'controller.json',state);event.set();thread.join(timeout=6);report()


if __name__=='__main__':main()
