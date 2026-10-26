"""Display the paired50 test-image scores in original acquisition-frame order."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/kinematic_resplit/analysis/psnr_distribution_final50k'


def main():
    with (OUT/'per_frame.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows = sorted(csv.DictReader(stream),key=lambda r:int(r['original_frame_id']))
    ids = np.array([int(r['original_frame_id']) for r in rows])
    if ids.tolist()!=list(range(1,201,4)):
        raise ValueError('Expected the fixed50 official test frames in acquisition order')
    values = {name:np.array([float(r[name+'_psnr_db']) for r in rows]) for name in ('V3','V4')}
    for model,folder in [('V3','V3_frame_time'),('V4','V4_frame_time_kinematic')]:
        path = ROOT/'runs/pidg_kinematic_resplit'/folder/'rubber_duck/per_view.json'
        original = json.loads(path.read_text())['ours_50000']['PSNR']
        if any(float(r[model+'_psnr_db'])!=original[r['render_file']] for r in rows):
            raise ValueError('Sorted CSV differs from source metrics')
    delta = values['V4']-values['V3']
    if not np.isfinite(np.r_[values['V3'],values['V4'],delta]).all():
        raise ValueError('Nonfinite PSNR')
    fig,(ax,diff) = plt.subplots(2,1,figsize=(15,7.5),sharex=True,
        gridspec_kw={'height_ratios':[3,1.2]},layout='constrained')
    for name,color in [('V3','#277da8'),('V4','#dc7a28')]:
        ax.plot(ids,values[name],'o-',ms=4,lw=1.7,color=color,label=name)
    ax.set(title='V3 / V4 test PSNR by original frame ID | resplit | final50k',ylabel='PSNR (dB)')
    ax.legend(ncol=2)
    ax.set_ylim(29.5,49.5)
    low = int(np.argmin(values['V3']))
    ax.annotate(f'Frame {ids[low]}\nV3 {values["V3"][low]:.3f} / V4 {values["V4"][low]:.3f} dB',
        xy=(ids[low],values['V3'][low]),xytext=(ids[low]-35,31.5),fontsize=10,
        arrowprops={'arrowstyle':'->','color':'#555555'})
    diff.bar(ids,delta,width=2.7,color=np.where(delta>=0,'#2d9276','#c34a56'))
    diff.axhline(0,lw=.8,color='#444444')
    diff.set(ylabel='V4 - V3 (dB)',xlabel='Original frame ID (ascending; only the 50 TEST frames)',
             xticks=ids,xlim=(-1,199))
    diff.tick_params(axis='x',labelrotation=90,labelsize=8)
    for axis in (ax,diff):
        axis.grid(axis='y',alpha=.2); axis.set_axisbelow(True)
        axis.spines['top'].set_visible(False); axis.spines['right'].set_visible(False)
    fig.savefig(OUT/'psnr_by_frame.png',dpi=180,facecolor='white')
    plt.close(fig)
    lines = ['# V3 / V4：按原始帧编号排列的测试集 PSNR','',
        '使用重新划分后的120 train /30 validation /50 test，两模型均取最终50k checkpoint。',
        '下表按原始帧号从小到大排列，仅包含测试帧；未计算的训练/验证帧不填充或插值。',
        'PSNR 为全图指标，单位 dB。Δ = V4 − V3。渲染文件编号从0开始，与原始数据帧编号不同。','',
        '![按原始帧编号的 PSNR](psnr_by_frame.png)','',
        '| 原始帧编号 | 渲染文件 | 归一化时间 | V3 PSNR | V4 PSNR | Δ PSNR |',
        '|---:|---|---:|---:|---:|---:|']
    for r,d in zip(rows,delta):
        lines.append(f'| {int(r["original_frame_id"])} | {r["render_file"]} | '
            f'{float(r["normalized_time"]):.6f} | {float(r["V3_psnr_db"]):.4f} | '
            f'{float(r["V4_psnr_db"]):.4f} | {d:+.4f} |')
    lines += ['','[下载逐帧 CSV](per_frame.csv) · [分布统计](summary.md)']
    (OUT/'psnr_by_frame.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'count':len(rows),'frame_order':ids.tolist(),
                      'plot':str(OUT/'psnr_by_frame.png'),'table':str(OUT/'psnr_by_frame.md')},indent=2))


if __name__=='__main__': main()
