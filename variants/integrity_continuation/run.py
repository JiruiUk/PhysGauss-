"""Resume exact existing states, preserve prior selection outcomes, and compare all models."""
import argparse
from datetime import datetime,timedelta,timezone
import html
import os
from pathlib import Path
import shutil
import sys
import threading
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.integrity_tuning import run as prior
from variants.integrity_feedback import run as feedback
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
HERE=Path(__file__).parent;RUNS=ROOT/'runs/pidg_integrity_continuation';REPORTS=ROOT/'experiments/integrity_continuation'
JOBS=[{'id':name,'run':root/name/'rubber_duck','repo':repo} for names,root,repo in (
    (['H1_balanced_coverage','H4_color_density'],prior.RUNS,prior.HERE/'code'),
    (['F1_feedback','F2_conservative_retry'],feedback.RUNS,feedback.HERE/'code')) for name in names]
REFERENCES={'C1_reused':prior.REFERENCE,'H2_temporal_cohesion':prior.RUNS/'H2_temporal_cohesion/rubber_duck',
    'H3_combined':prior.RUNS/'H3_combined/rubber_duck'}


def verify_checkpoint(directory):
    path=directory/'checkpoints/resume.pt';manifest=engine.read_json(path.with_suffix('.json'))
    digest=engine.sha256(path)
    if digest!=manifest['sha256'] or not 0<manifest['iteration']<=50000:raise ValueError('Invalid continuation checkpoint')
    if not manifest.get('optimizer_saved') or not manifest.get('rng_saved'):raise ValueError('Full optimizer/RNG checkpoint required')
    sidecar=engine.read_json(path.with_suffix('.metrics.json'));config=engine.read_json(directory/'metric_config.json')
    if sidecar['checkpoint_sha256']!=digest or sidecar['iteration']!=manifest['iteration'] or sidecar['config']!=config:
        raise ValueError('Continuation config/selection state differs from checkpoint')
    for key,suffix in [('coverage_sha256','.coverage.pt'),('color_evidence_sha256','.color.pt')]:
        if config.get('coverage_density') and engine.sha256(path.with_suffix(suffix))!=sidecar[key]:
            raise ValueError('Continuation density sidecar differs')
    return {'iteration':manifest['iteration'],'sha256':digest,'config_sha256':engine.sha256(directory/'metric_config.json'),
        'model_config_sha256':engine.sha256(directory/'model_config.py'),'needs_training':manifest['iteration']<50000}


def freeze():
    files=[*HERE.glob('*.py'),HERE/'launch.ps1',ROOT/'tests/test_integrity_continuation.py']
    files += [j['run']/name for j in JOBS for name in ('model_config.py','metric_config.json','branch.json')]
    return {'prior':prior.freeze(prior.HERE/'code'),'feedback':feedback.freeze(feedback.HERE/'code'),
        'helpers_and_recipes':{str(p.relative_to(ROOT)):engine.sha256(p) for p in files}}


def assert_frozen(repo=None):
    prior.assert_frozen(prior.HERE/'code');feedback.assert_frozen(feedback.HERE/'code')
    if engine.read_json(REPORTS/'source_freeze.json')!=freeze():raise ValueError('Continuation source or recipe changed')


def wire():
    engine.RUNS,engine.REPORTS,engine.assert_frozen=RUNS,REPORTS,assert_frozen
    base.RUNS,base.REPORTS,base.assert_frozen=RUNS,REPORTS,assert_frozen


def archive_before_resume(job):
    destination=RUNS/'resume_sources'/job['id'];receipt=destination/'archive.json'
    if receipt.exists():return
    source=job['run'];verified=verify_checkpoint(source)
    for name in ('checkpoints','point_cloud','deform'):
        shutil.copytree(source/name,destination/name,dirs_exist_ok=True)
    for name in ('model_config.py','metric_config.json','metric_best.json','metric_history.jsonl','integrity_history.jsonl','branch.json','screen.json'):
        if (source/name).exists():shutil.copyfile(source/name,destination/name)
    if engine.sha256(destination/'checkpoints/resume.pt')!=verified['sha256']:raise ValueError('Pre-continuation archive differs')
    engine.write_json(receipt,{'source':str(source),'iteration':verified['iteration'],'full_checkpoint_sha256':verified['sha256'],
        'selected_snapshots_preserved':True,'created_utc':engine.now()})


def report():
    state=engine.read_json(RUNS/'controller.json') if (RUNS/'controller.json').exists() else {}
    active=engine.read_json(RUNS/'active.json') if (RUNS/'active.json').exists() else {}
    rows=[]
    for job in JOBS:
        d=job['run'];history=prior.read_history(d);iteration=base.progress(d)
        rows.append({'id':job['id'],'iteration':iteration,'source':str(d),'history':history,
            'status':'completed' if iteration>=50000 else 'training' if active.get('id')==job['id'] and state.get('status')=='training' else 'queued',
            'validation':history[-1] if history else None})
    result={'updated_utc':engine.now(),'controller':state,'active':active,'rows':rows,
        'budget':engine.read_json(REPORTS/'budget.json'),'baseline_retrained':False,
        'reason':'User explicitly requested full50k completion despite earlier20k pilot rejection; prior decisions preserved.'}
    engine.write_json(REPORTS/'results.json',result)
    lines=['# Requested continuation to50k','',
        'Status: '+state.get('status','preparing')+' | Active: '+active.get('id','-')+' / '+active.get('phase','-'),
        '','H1/H4/F1/F2 resume existing20k full states. C1/H2/H3 reused; no original recipe changed.',
        'Prior pilot rejection is preserved; this continuation is an explicitly requested ablation.',
        '','| Model | Iter | State | Validation PSNR | Integrity score |','|---|---:|---|---:|---:|']
    for r in rows:
        v=r['validation'] or {};lines.append(f"| {r['id']} | {r['iteration']}/50000 | {r['status']} | {v.get('metrics',{}).get('PSNR','-')} | {v.get('integrity',{}).get('integrity_score','-')} |")
    if state.get('error'):lines+=['','Error: '+state['error']]
    engine.write_text_atomic(REPORTS/'results.md','\n'.join(lines)+'\n')
    page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="5"><title>50k continuation</title><style>body{font:16px system-ui;background:#142033;color:#edf5ff;margin:24px}section,svg{background:#203047;padding:12px;margin:12px 0}progress{width:90%}a{color:#8de4b1}</style><h1>未完成模型续训 / 50k</h1><p>'+html.escape(state.get('status','preparing')+' | '+active.get('id','-')+' / '+active.get('phase','-'))+'</p><p>Updated UTC: '+result['updated_utc']+' | Deadline UTC: '+result['budget']['deadline_utc']+'</p><p>恢复完整 optimizer/RNG 状态；C1、H2、H3 复用。先前筛选结论保留，续训不表示已验证提升。</p>'
    if state.get('error'):page+='<section style="border:2px solid #ff7979"><pre>'+html.escape(state['error'])+'</pre></section>'
    for r in rows:page+=f"<section><b>{r['id']} / {r['status']}</b><h2>{r['iteration']:,} / 50,000 iter</h2><progress value='{r['iteration']}' max='50000'></progress></section>"
    for label,key in [('PSNR','PSNR'),('SSIM','SSIM'),('LPIPS (lower)','LPIPS'),('MS-SSIM','MS-SSIM'),('Integrity score (lower)','integrity_score'),('Interior low50','core_low_50'),('Interior low90','core_low_90'),('Fragment loss','fragment_loss'),('Opaque appearance error','opaque_appearance_error'),('Background spill','background_false_coverage')]:
        values=[h['metrics'][key] if key in engine.METRICS else h['integrity'][key] for r in rows for h in r['history']]
        if not values:continue
        lo,hi=min(values),max(values);span=max(hi-lo,1e-8)
        page+=f'<h2>{label}</h2><svg viewBox="0 0 1000 290" width="1000"><text x="5" y="20" fill="white">{hi:.6f}</text><text x="5" y="240" fill="white">{lo:.6f}</text>'
        for index,(r,color) in enumerate(zip(rows,['#86d1ff','#8de4b1','#ffbe83','#e9a0ee'])):
            points=' '.join(f"{85+880*h['iteration']/50000:.1f},{240-200*((h['metrics'][key] if key in engine.METRICS else h['integrity'][key])-lo)/span:.1f}" for h in r['history'])
            page+=f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="{65+index*230}" y="275" fill="{color}">{r["id"]}</text>'
        page+='</svg>'
    engine.write_text_atomic(REPORTS/'monitor.html',page+'<p><a href="results.md">Results</a> | <a href="../training_monitor/index.html">Earlier experiments</a></p>')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=8.);parser.add_argument('--renew-budget',action='store_true');args=parser.parse_args()
    if args.hours<=0:raise ValueError('Positive continuation budget required')
    RUNS.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    prior.assert_frozen(prior.HERE/'code');feedback.assert_frozen(feedback.HERE/'code')
    if (REPORTS/'source_freeze.json').exists():assert_frozen()
    else:engine.write_json(REPORTS/'source_freeze.json',freeze())
    budget_path=REPORTS/'budget.json'
    if args.renew_budget or not budget_path.exists():
        if budget_path.exists():shutil.copyfile(budget_path,REPORTS/('budget_before_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'.json'))
        now=datetime.now(timezone.utc);engine.write_json(budget_path,{'started_utc':now.isoformat(),'deadline_utc':(now+timedelta(hours=args.hours)).isoformat(),'hours':args.hours,'source':'explicit user continuation request'})
    deadline=datetime.fromisoformat(engine.read_json(budget_path)['deadline_utc'])
    audit={j['id']:verify_checkpoint(j['run']) for j in JOBS}
    engine.write_json(REPORTS/'resume_verification.json',{'checked_utc':engine.now(),'jobs':audit,'exact_saved_configs_required':True})
    if not (REPORTS/'plan.json').exists():
        engine.write_json(REPORTS/'plan.json',{'jobs':[{**{k:str(v) for k,v in j.items()},'start':audit[j['id']]} for j in JOBS],
            'target_total':50000,'reused':{k:str(v) for k,v in REFERENCES.items()},'user_overrode_pilot_filter':True,'split':[120,30,50]})
    for label,p in [('integrity_selection',prior.REPORTS/'selection.json'),('feedback_reviews',feedback.REPORTS/'screen_reviews.json'),('feedback_proposals',feedback.REPORTS/'proposals.json')]:
        if not (REPORTS/(label+'_before_continuation.json')).exists():
            shutil.copyfile(p,REPORTS/(label+'_before_continuation.json'))
    wire();state={'pid':os.getpid(),'status':'training','started_utc':engine.now(),'deadline_utc':deadline.isoformat(),'completed':[]}
    event=threading.Event()
    def monitor():
        while not event.wait(5):
            try:report()
            except (OSError,ValueError,KeyError):pass
    with engine.lock():
        engine.write_json(RUNS/'controller.json',state);thread=threading.Thread(target=monitor,daemon=True);thread.start();report()
        os.environ['PIDG_EXPORT_ALPHA']='1'
        try:
            for job in JOBS:
                if datetime.now(timezone.utc)>=deadline:raise SystemExit(75)
                d=job['run'];engine.repo_global=job['repo'];verified=verify_checkpoint(d)
                if verified['needs_training']:
                    archive_before_resume(job)
                    code=base.train(d,job['repo'],deadline)
                    if code==75:raise SystemExit(75)
                    if code:raise RuntimeError('Continuation failed: '+job['id'])
                state['completed'].append(job['id']);engine.write_json(RUNS/'controller.json',state)
            engine.write_json(RUNS/'selection_lock.json',{'locked_utc':engine.now(),'reason':'All requested variants completed50k; selections use validation only',
                'validation_selections':{j['id']:engine.read_json(j['run']/'metric_best.json') for j in JOBS},
                'integrity_selections':{j['id']:engine.read_json(j['run']/'integrity_best.json') if (j['run']/'integrity_best.json').exists() else None for j in JOBS}})
            state['status']='final_test';engine.write_json(RUNS/'controller.json',state)
            for job in JOBS:
                if datetime.now(timezone.utc)>=deadline:raise SystemExit(75)
                engine.repo_global=job['repo'];d=job['run']
                tags=['50000','best']+(['integrity_best'] if (d/'integrity_best.json').exists() else [])
                base.render(d,job['repo'],deadline,tags=tags)
            from .export import export_and_assess
            export_and_assess();state['status']='completed';assert_frozen()
        except SystemExit as exc:
            state['status']='saved_stopped' if exc.code==75 else 'failed'
            if exc.code!=75:raise
        except Exception as exc:
            state.update(status='failed',error=str(exc));raise
        finally:
            state['finished_utc']=engine.now();engine.write_json(RUNS/'controller.json',state);event.set();thread.join(timeout=6);report()


if __name__=='__main__':main()
