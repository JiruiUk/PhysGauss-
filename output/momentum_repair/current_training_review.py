"""CPU-only live review of completed validation events, never load the GPU."""
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.momentum_guard.run import assert_frozen
assert_frozen(ROOT/'variants/momentum_guard/code')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
run=ROOT/'runs/pidg_momentum_guard/M1_guarded_shared/rubber_duck'
control=ROOT/'runs/pidg_unified_flow/L1_shared_stress/rubber_duck'
out=Path(__file__).parent
h=rows(run/'metric_history.jsonl');ch=rows(control/'metric_history.jsonl')
latest=h[-1];same=next(r for r in ch if r['iteration']==latest['iteration'])
best=read(run/'metric_best.json')['PSNR'];cbest=read(control/'metric_best.json')['PSNR']
guard=rows(run/'momentum_gradient_guard.jsonl')
for r in guard:
    assert 0<=r['scale']<=1 and r['private_gradients_preserved']
    assert all(g['guarded_momentum_norm']<=.1*g['image_norm']*(1+1e-5) for g in r['groups'].values())
momentum=rows(run/'material_momentum.jsonl');step=latest['iteration']
window=[r for r in momentum if step-3000<r['iteration']<=step]
cwindow=[r for r in rows(control/'material_momentum.jsonl') if step-3000<r['iteration']<=step]
early={r['iteration']:r['metrics']['PSNR'] for r in ch if r['iteration']<=8000}
pre=[abs(r['metrics']['PSNR']-early[r['iteration']]) for r in h if r['iteration'] in early]
summary={'review_utc':datetime.now(timezone.utc).isoformat(),'validation_step':step,
    'latest':latest,'same_step_control':same,'current_best':best,'completed_control_best':cbest,
    'latest_minus_control_best_PSNR':latest['metrics']['PSNR']-cbest['value'],
    'LPIPS_reduction_vs_control_best':1-latest['metrics']['LPIPS']/cbest['metrics']['LPIPS'],
    'relative_MSE_reduction_from_PSNR':1-10**(-(latest['metrics']['PSNR']-cbest['value'])/10),
    'pre_physics_max_abs_PSNR_difference':max(pre),'guard_events_checked':len(guard),
    'all_guard_caps_valid':True,'last_guard':guard[-1],
    'last_3k_raw_huber_mean':{'M1':float(np.mean([r['raw_huber'] for r in window])),
        'L1_same_iteration_window':float(np.mean([r['raw_huber'] for r in cwindow]))},
    'limits':'Training in progress; no new test opened or independent 2D evaluation yet. Different point sets and local neighborhoods make raw residual probe averages non-identical samples; finite-difference residual is not physical GT.'}
(out/'current_training_review.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
for hist,label,color in [(ch,'L1 completed control','#c1544e'),(h,'M1 gradient guard (in progress)','#267eae')]:
    for ax,key in zip(axes[:2],['PSNR','LPIPS']):
        ax.plot([r['iteration'] for r in hist],[r['metrics'][key] for r in hist],label=label,color=color)
for ax,key in zip(axes[:2],['PSNR (dB), higher better','LPIPS, lower better']):
    ax.axvline(8100,color='gray',linestyle='--',alpha=.6);ax.set(xlabel='Total training iterations',ylabel=key)
axes[0].set_ylim(28,41);axes[1].set_ylim(0,.06)
for folder,label,color in [(control,'L1 raw momentum','#c1544e'),(run,'M1 raw momentum','#267eae')]:
    r=rows(folder/'material_momentum.jsonl')
    axes[2].plot([v['iteration'] for v in r],[v['raw_huber'] for v in r],alpha=.65,color=color,label=label)
axes[2].set(xlabel='Total training iterations',ylabel='Unweighted Huber residual (log)',yscale='log')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle(f'Interim review at validation {step:,}: image gains are not physical validation')
fig.savefig(out/'current_training_review.png',dpi=160);plt.close(fig)
print(json.dumps(summary,ensure_ascii=False))
