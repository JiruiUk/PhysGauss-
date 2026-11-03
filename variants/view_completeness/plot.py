"""Scientific plot of the fixed, target-free thin-support descriptor."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'experiments/view_completeness'
OUT = ROOT/'output/view_completeness'
OUT.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
for ax, name in zip(axes, ['C1_original_50k', 'T1_association_52k', 'O1_flow_on_50k']):
    records = json.loads((REPORT/f'{name}_metrics.json').read_text(encoding='utf-8'))
    for view, color in [('moving_control', '#555555'), ('fixed_0002', '#0072B2'),
                        ('fixed_0086', '#D55E00'), ('fixed_0188', '#009E73')]:
        rows = records[view]
        ax.plot([r['frame'] for r in rows], [100*r['shape_proxies']['thin_support_fraction'] for r in rows],
                color=color, label=view, linewidth=1.3, alpha=.9)
    ax.set_title(name, loc='left')
    ax.set_ylabel('Thin support / foreground (%)')
    ax.grid(alpha=.2)
axes[0].legend(ncol=4, fontsize=9)
axes[-1].set_xlabel('Original global time frame (1-200)')
fig.suptitle('Fixed-camera temporal shape descriptor — proxy, not true completeness', fontsize=13)
fig.savefig(OUT/'thin_support_by_time.png', dpi=160)
fig.savefig(OUT/'thin_support_by_time.pdf')
