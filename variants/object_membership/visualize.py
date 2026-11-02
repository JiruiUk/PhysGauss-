"""Static scientific figures from verified membership arrays; no model loads."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'experiments/object_membership'
OUT = ROOT / 'output/object_membership'
COLORS = ['#ec4899', '#06b6d4', '#22c55e', '#f59e0b']


def main():
    r = json.loads((REPORT / 'results.json').read_text())
    receipt = json.loads((REPORT / 'verification.json').read_text())
    assert receipt['passed']
    with np.load(OUT / 'membership.npz') as z:
        data = {k: z[k] for k in z.files}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#f8fafc', 'axes.facecolor': '#ffffff'})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.6), constrained_layout=True)
    for ax, (domain, n) in zip(axes.flat, [('early_2_27', 6), ('early_2_27', 14), ('early_2_27', 27),
                                          ('late_188_200', 192), ('late_188_200', 195), ('late_188_200', 200)]):
        t = data[domain + '_frames'].tolist().index(n)
        labels = data[domain + '_visible_owner'][t]
        with Image.open(ROOT / f'output/short_clip_diagnostic/native_gt_render/frame{n:04d}.png') as im:
            rgb = np.array(im.crop((0, 0, 1600, 900)).resize((400, 225), Image.Resampling.LANCZOS))
        with np.load(ROOT / f'runs/pidg_short_clip_diagnostic/cache/frame{n:04d}.npz') as z:
            means = z['means']
        ax.imshow(rgb)
        for k, color in enumerate(COLORS):
            points = means[labels == k]
            ax.scatter(points[:, 0], points[:, 1], s=2.2, color=color, alpha=.65, linewidths=0)
        ax.set_xlim(0, 399)
        ax.set_ylim(224, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        row = next(d for d in r['domains'] if d['domain'] == domain)['frames'][t]
        ax.set_title(f"Frame {n} | {row['currently_visible_member_points']:,} visible members | alpha {100*row['current_visible_member_alpha_coverage']:.1f}%", fontsize=10)
        ax.set_xlabel('Early local IDs (continuous 2-27)' if domain.startswith('early') else 'Late local IDs (independent; no early/late match)', fontsize=9)
    fig.suptitle('Frozen C1: Gaussian centers supported by local object evidence\nColors apply only within each row; unassigned points are omitted', fontsize=14)
    fig.legend(handles=[Line2D([], [], marker='o', linestyle='', color=c, label=f'Local F{k}') for k, c in enumerate(COLORS)],
               loc='outside lower center', ncol=4, frameon=False)
    fig.savefig(OUT / 'point_objects.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    for domain, color, label in [(r['domains'][0], '#2563eb', 'Early'), (r['domains'][1], '#a855f7', 'Late')]:
        xx = [v['frame'] for v in domain['frames']]
        axes[0].plot(xx, [v['historical_member_points'] for v in domain['frames']], color=color, label=label + ' historical')
        axes[0].plot(xx, [v['currently_visible_member_points'] for v in domain['frames']], color=color, linestyle='--', label=label + ' visible')
    axes[0].set(xlabel='Original training frame', ylabel='Gaussian rows', title='Association and visibility are separate')
    axes[0].legend(fontsize=8, frameon=False)
    matrix = np.array([b['row_overlap_with_A2_objects'] for b in r['patch_bridges']])
    axes[1].imshow(matrix, cmap='Blues')
    axes[1].set(xticks=range(4), yticks=range(4), xticklabels=[f'A2 F{k}' for k in range(4)],
                yticklabels=[f'A15 F{k}' for k in range(4)], title='Row overlap after image-link verification')
    for (i, j), v in np.ndenumerate(matrix):
        axes[1].text(j, i, str(v), ha='center', va='center', color='white' if v > matrix.max() / 2 else '#0f172a')
    xx = [v['frame'] for v in r['causal_bank_comparison_after_A15_seeds']]
    for mode, label, color in [('A2_only', 'A2 patch only', '#64748b'), ('A2_plus_A15', 'A2 + verified A15 patch', '#2563eb')]:
        yy = [100*v[mode]['accepted_alpha_coverage'] for v in r['causal_bank_comparison_after_A15_seeds']]
        axes[2].plot(xx, yy, 'o-', label=label, color=color)
    axes[2].set(xlabel='Later frame (A15 selection ended at 19)', ylabel='Supported duck alpha (%)',
                title='Same per-point gates, causal bank addition', ylim=(55, 72))
    axes[2].legend(fontsize=8, frameon=False)
    fig.suptitle('Conditional local associations; these are not physical identity accuracy scores', fontsize=13)
    fig.savefig(OUT / 'evidence.png', dpi=150)
    plt.close(fig)
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>C1 点与局部物体归属</title>
<style>body{max-width:1450px;margin:30px auto;padding:0 24px;font:17px system-ui;background:#f8fafc;color:#172033}img{width:100%;margin:12px 0;border:1px solid #dbe3ef;border-radius:8px}p{line-height:1.7}</style>
<h1>点可以先归到局部物体，再单独验证材料表面身份</h1>
<p>原2–27的四条连续可见图像对应经正反光流核验；A15四点库分别连至A2四个局部物体。不同点库不必是同一块表面。
晚期188–200独立编号，与早期相同颜色不表示同一只鸭子。图中仅绘出当前有贡献/中心/运动支持且已积累连续三观测证据的Gaussian中心。</p>
<img src="point_objects.png" alt="局部物体与Gaussian中心"><img src="evidence.png" alt="历史与可见归属、点库重叠和因果补充对照">
<p>原C1未重训；使用1600实际渲染完整贡献缓存、400图像掩膜和中心/流坐标。覆盖率不是物理身份准确率。未知、遮挡和跨物体冲突保留未知。
本诊断逐点规则与先前整库拒绝规则不同，不能直接将两轮覆盖率相减作为模型改进。</p>
<p><a href="../../experiments/object_membership/assessment.md">判断与下一步</a> · <a href="../../experiments/object_membership/results.json">数值</a> ·
<a href="../../experiments/object_membership/verification.json">独立核验</a></p></html>'''
    (OUT / 'index.html').write_text(html, encoding='utf-8')
    print('Membership plots and gallery written', flush=True)


if __name__ == '__main__':
    main()
