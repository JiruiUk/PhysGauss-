"""Bounded, resumable50k ablations; H3 reused; validation-only choices and locked test."""
import argparse
from datetime import datetime,timedelta,timezone
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.integrity_tuning import run as prior
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from variants.centroid_tuning import run as coordinator
from .prepare import prepare
HERE=Path(__file__).parent;REPORTS=ROOT/'experiments/temporal_repair';RUNS=ROOT/'runs/pidg_temporal_repair'
NAMES=['R1_unit_rotation','R2_hard_train_frames','R3_combined']
REFERENCE=prior.RUNS/'H3_combined/rubber_duck'


def configuration(name,smoke=False,off=False):
    c,text=prior.configuration('H3_combined',smoke)
    c.update(integrity_unit_rotation=name in ('R1_unit_rotation','R3_combined','R0_smoke') and not off,
        integrity_hard_frame_probability=.25 if name in ('R2_hard_train_frames','R3_combined','R0_smoke') and not off else 0.,
        integrity_hard_frame_fraction=.25,integrity_hard_frame_start=20 if smoke else 8000)
    return c,text


def freeze(repo):
    files=[*HERE.glob('*.py'),ROOT/'tests/test_temporal_repair.py',ROOT/'tests/test_temporal_repair_gpu.py',REPORTS/'budget.json',REPORTS/'plan.json']
    files+=[REFERENCE/f for f in ('model_config.py','metric_config.json','metric_history.jsonl','per_view.json','integrity_history.jsonl','checkpoints/resume.json')]
    return {'parent':prior.freeze(prior.HERE/'code'),'code':engine.runtime_hashes(repo),
        'inputs':{str(p.relative_to(ROOT)):engine.sha256(p) for p in files}}


def assert_frozen(repo):
    if engine.read_json(REPORTS/'source_freeze.json')!=freeze(repo):raise ValueError('Temporal repair source/config changed')


def wire(repo):
    engine.RUNS,engine.REPORTS,engine.repo_global,engine.assert_frozen=RUNS,REPORTS,repo,assert_frozen
    base.RUNS,base.REPORTS,base.assert_frozen=RUNS,REPORTS,assert_frozen
    coordinator.RUNS,coordinator.REPORTS,coordinator.configuration=RUNS,REPORTS,configuration
    coordinator.BASE='C1_repaired'


def report():
    state=engine.read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {}
    active=engine.read_json(RUNS/'active.json') if (RUNS/'active.json').exists() else {}
    rows=[]
    for name in NAMES:
        d=RUNS/name/'rubber_duck';history=prior.read_history(d)
        rows.append({'id':name,'iteration':base.progress(d),'history':history,'validation':history[-1] if history else None})
    progress={'iteration':0,'target':100 if '_smoke' in active.get('run','') else 50000}
    if active.get('run'):
        d=Path(active['run']);matches=re.findall(r'Training progress[^\r\n]*?\|\s*(\d+)/(\d+)',engine.tail(d/'train.log',100000))
        if matches:progress.update(iteration=int(matches[-1][0]),target=int(matches[-1][1]))
        if (d/'checkpoints/resume.json').exists():progress['iteration']=max(progress['iteration'],coordinator.step(d))
    result={'updated_utc':engine.now(),'controller':state,'active':active,'active_progress':progress,'rows':rows,
        'baseline_retrained':False,'reference':str(REFERENCE),'budget':engine.read_json(REPORTS/'budget.json')}
    engine.write_json(REPORTS/'results.json',result)
    text=['# H3 worst-frame repair ablations','',f"Status: {state.get('status','preparing')} | Active: {active.get('id','-')} | {progress['iteration']}/{progress['target']} iter",'',
        'H3 reused. R1 unit covariance rotation; R2 training-error hard-frame sampling; R3 both.',
        'Shared random-training3k full states, total50k each;120 train/30 validation/50 fixed test.',
        'No test/validation RGB or alpha supervises these changes. Validation chooses; historical test exposure remains.','',
        '| Model | Iter | Validation PSNR | Integrity score |','|---|---:|---:|---:|']
    for r in rows:
        v=r['validation'] or {};text.append(f"| {r['id']} | {r['iteration']} | {v.get('metrics',{}).get('PSNR','-')} | {v.get('integrity',{}).get('integrity_score','-')} |")
    if state.get('error'):text+=['','Error: '+state['error']]
    engine.write_text_atomic(REPORTS/'results.md','\n'.join(text)+'\n')
    page='<meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>H3 repair</title><style>body{font:16px system-ui;background:#142033;color:#edf5ff;margin:24px}section{background:#203047;padding:16px;margin:12px 0}svg{background:#203047;max-width:100%}a{color:#8de4b1}pre{white-space:pre-wrap}</style><h1>H3 最差帧修正 / 受控消融</h1>'
    page+=f"<section><h2>{html.escape(state.get('status','preparing'))} | {html.escape(active.get('id','-'))}</h2><h2>当前 {progress['iteration']:,} / {progress['target']:,} iter</h2><p>每5秒刷新；UTC {result['updated_utc']}；预算截至 {result['budget']['deadline_utc']}</p></section>"
    if state.get('error'):page+='<section style="border:2px solid #f88">'+html.escape(state['error'])+'</section>'
    for r in rows:page+=f"<section>{r['id']}<h2>{r['iteration']:,} / 50,000 iter</h2></section>"
    reference=prior.read_history(REFERENCE)
    for label,key in [('Validation PSNR','PSNR'),('Validation LPIPS (lower)','LPIPS'),('Integrity score (lower)','integrity_score'),('Core low90 (lower)','core_low_90'),('Background spill (lower)','background_false_coverage')]:
        series=[('H3 reused',reference)]+[(r['id'],r['history']) for r in rows]
        values=[h['metrics'][key] if key in engine.METRICS else h['integrity'][key] for _,s in series for h in s]
        if not values:continue
        lo,hi=min(values),max(values);span=max(hi-lo,1e-8);page+='<h3>'+label+'</h3><svg viewBox="0 0 1000 280" width="1000">'
        page+=f'<text x="4" y="20" fill="white">{hi:.6f}</text><text x="4" y="220" fill="white">{lo:.6f}</text>'
        for j,((name,s),color) in enumerate(zip(series,['#aaa','#86d1ff','#8de4b1','#ffbe83'])):
            points=' '.join(f"{80+880*h['iteration']/50000:.1f},{220-190*((h['metrics'][key] if key in engine.METRICS else h['integrity'][key])-lo)/span:.1f}" for h in s)
            page+=f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="{70+j*220}" y="260" fill="{color}">{name}</text>'
        page+='</svg>'
    page+='<pre>'+html.escape('\n'.join(text))+'</pre><a href="diagnosis.md">最差帧诊断</a> | <a href="readout_validation.json">插值诊断</a> | <a href="assessment.md">完成后比较</a>'
    engine.write_text_atomic(REPORTS/'monitor.html',page)


def preflight(repo,deadline):
    env=engine.env_for(RUNS);env['PIDG_TEMPORAL_GPU_TEST']='1'
    check=subprocess.run([sys.executable,'-m','pytest','tests/test_temporal_repair.py','tests/test_temporal_repair_gpu.py','-q'],cwd=ROOT,env=env,capture_output=True)
    (REPORTS/'tests.log').write_bytes(check.stdout+check.stderr)
    if check.returncode:raise RuntimeError('Temporal repair unit/CUDA tests failed')
    d=coordinator.initialize('R0_smoke',True)
    prior.branch(prior.RUNS/'_smoke/shared_prefix/rubber_duck',d,20)
    if coordinator.step(d)<70 and base.train(d,repo,deadline,True,limit=70)!=75:raise ValueError('Smoke save boundary failed')
    if coordinator.step(d)<100 and base.train(d,repo,deadline,True)!=0:raise ValueError('Smoke training failed')
    if b'FULL_CHECKPOINT_RESTORED completed=70 next=71' not in (d/'train.log').read_bytes():raise ValueError('Smoke resume not exercised')
    sidecar=engine.read_json(d/'checkpoints/resume.metrics.json')
    if not sidecar.get('frame_sampler',{}).get('ema'):raise ValueError('Training error sampler was not checkpointed')
    engine.write_json(REPORTS/'preflight.json',{'status':'passed','smoke_iterations':100,'resume_boundary':70,
        'flags_off_pixel_identity':True,'unit_rotation_finite_renderer_gradients':True,'frame_sampler_checkpointed':True})
    paths=['variants/temporal_repair','tests/test_temporal_repair.py','tests/test_temporal_repair_gpu.py',
        *['experiments/temporal_repair/'+f for f in ['plan.json','budget.json','diagnosis.json','diagnosis.md','source_freeze.json','preflight.json','tests.log']]]
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True)
    result=subprocess.run(['git','commit','--only','-m','Add validated H3 worst-frame repair ablations','--',*paths],cwd=ROOT,capture_output=True)
    (REPORTS/'commit.log').write_bytes(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Repair milestone commit failed')


def choose():
    reference=prior.read_history(REFERENCE)[-1];path=REPORTS/'readout_validation.json'
    baseline_worst=engine.read_json(path)['models']['native']['worst5_PSNR']
    reviews=[]
    for name in NAMES:
        d=RUNS/name/'rubber_duck';r=prior.read_history(d)[-1]
        tail=[json.loads(s) for s in (d/'temporal_validation.jsonl').read_text().splitlines()][-1]
        a,b=r['metrics'],reference['metrics'];g,h=r['integrity'],reference['integrity']
        guards={'PSNR':a['PSNR']>=b['PSNR']-.05,'SSIM':a['SSIM']>=b['SSIM']-.0002,
            'LPIPS':a['LPIPS']<=b['LPIPS']+.0003,'MS-SSIM':a['MS-SSIM']>=b['MS-SSIM']-.0002,
            'worst5':tail['worst5_PSNR']>=baseline_worst+.2,'coverage':g['core_low_90']<=h['core_low_90']*1.1,
            'spill':g['background_false_coverage']<=h['background_false_coverage']+.001}
        reviews.append({'model':name,'validation':r,'worst5_PSNR':tail['worst5_PSNR'],'guards':guards,'eligible':all(guards.values())})
    eligible=[r for r in reviews if r['eligible']]
    selected=max(eligible,key=lambda r:r['validation']['metrics']['PSNR'])['model'] if eligible else 'H3_reused'
    engine.write_json(REPORTS/'selection.json',{'selection_split':'validation','selected':selected,'reviews':reviews,
        'no_forced_improvement':True,'historical_test_exposure':True})
    engine.write_json(RUNS/'selection_lock.json',{'locked_utc':engine.now(),'selection_split':'validation','selections':{
        n:engine.read_json(RUNS/n/'rubber_duck/metric_best.json') for n in NAMES},'selected':selected})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=8.);args=parser.parse_args()
    RUNS.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    if args.hours<=0:raise ValueError('Positive bounded budget required')
    if not (REPORTS/'budget.json').exists():
        now=datetime.now(timezone.utc);engine.write_json(REPORTS/'budget.json',{'started_utc':now.isoformat(),'deadline_utc':(now+timedelta(hours=args.hours)).isoformat(),'hours':args.hours})
    if not (REPORTS/'plan.json').exists():
        engine.write_json(REPORTS/'plan.json',{'models':NAMES,'reference':str(REFERENCE),'reference_retrained':False,
            'total_iterations':50000,'branch_iteration':3000,'split':[120,30,50],
            'new_parameters':False,'loss_targets':'training RGB/alpha only','selection':'validation only; preserve H3 if no eligible repair',
            'R1':'Normalize additive deformed quaternion before CUDA covariance; no new material or force',
            'R2':'25% replacement from top25% training foreground-L1 EMA frames, after8k; original stack supplies remaining75%',
            'R3':'R1 plus R2','interpolation_prior':'not enabled; simple interpolation worsened validation frame84'})
    deadline=datetime.fromisoformat(engine.read_json(REPORTS/'budget.json')['deadline_utc'])
    prior.assert_frozen(prior.HERE/'code');repo=prepare()
    if not (REPORTS/'source_freeze.json').exists():engine.write_json(REPORTS/'source_freeze.json',freeze(repo))
    assert_frozen(repo);wire(repo)
    state={'pid':os.getpid(),'status':'preflight','started_utc':engine.now(),'completed':[]};stop=threading.Event()
    def sample():
        while not stop.wait(5):
            try:report()
            except (OSError,ValueError,KeyError):pass
    with engine.lock():
        engine.write_json(RUNS/'controller.json',state);thread=threading.Thread(target=sample,daemon=True);thread.start();report()
        try:
            if not (REPORTS/'preflight.json').exists():preflight(repo,deadline)
            for name in NAMES:
                if datetime.now(timezone.utc)>=deadline:raise SystemExit(75)
                state.update(status='training',active=name);engine.write_json(RUNS/'controller.json',state)
                d=coordinator.initialize(name);prior.branch(prior.parent.RUNS/'shared_prefix/rubber_duck',d,3000)
                if coordinator.step(d)<50000:
                    code=base.train(d,repo,deadline)
                    if code==75:raise SystemExit(75)
                state['completed'].append(name);engine.write_json(RUNS/'controller.json',state);report()
            choose();state['status']='locked_test_export';engine.write_json(RUNS/'controller.json',state)
            os.environ['PIDG_EXPORT_ALPHA']='1'
            for name in NAMES:base.render(RUNS/name/'rubber_duck',repo,deadline,split='test',tags=['50000'])
            from .export import export_and_assess
            export_and_assess();state.update(status='completed',finished_utc=engine.now())
        except SystemExit as error:
            state.update(status='saved_stopped',stop_code=error.code)
        except Exception as error:
            state.update(status='failed',error=str(error));raise
        finally:
            stop.set();thread.join(timeout=10);engine.write_json(RUNS/'controller.json',state);report()


if __name__=='__main__':main()
