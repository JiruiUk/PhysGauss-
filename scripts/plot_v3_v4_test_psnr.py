"""Read-only analysis of paired final50k PSNR on the fixed official50 test views."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/kinematic_resplit/analysis/psnr_distribution_final50k'
RUNS = ROOT/'runs/pidg_kinematic_resplit'
NAMES = {'V3':'V3_frame_time','V4':'V4_frame_time_kinematic'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stats(values):
    return {'count':len(values),'mean':float(np.mean(values)), 'median':float(np.median(values)),
            'population_std':float(np.std(values,ddof=0)),
            **{name:float(np.quantile(values,q)) for name,q in
               [('min',0),('p10',.1),('q1',.25),('q3',.75),('p90',.9),('max',1)]}}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    metadata = ROOT/'data/derived/kinematic_resplit/final_test/transforms_test.json'
    frames = read(metadata)['frames']
    ids = np.array([int(Path(f['file_path']).stem) for f in frames])
    expected = [f'{i:05d}.png' for i in range(50)]
    if len(frames)!=50 or ids.tolist()!=list(range(1,201,4)):
        raise ValueError('Official test membership/order changed')
    sources = [metadata]
    series = {}
    for name,folder in NAMES.items():
        p = RUNS/folder/'rubber_duck/per_view.json'; sources.append(p)
        per_view = read(p)['ours_50000']['PSNR']
        if set(per_view)!=set(expected): raise ValueError('Not exactly50 paired test metrics')
        values = np.array([per_view[k] for k in expected],dtype=np.float64)
        if not np.isfinite(values).all(): raise ValueError('Nonfinite PSNR')
        aggregate = read(p.parent/'results.json')['ours_50000']['PSNR']
        if abs(values.mean()-aggregate)>1e-5: raise ValueError('Per-frame mean disagrees with original evaluator')
        series[name] = values
    # Verify comparisons use identical rendered GT, not just identical file names.
    for image in expected:
        gt = [RUNS/folder/'rubber_duck/test/ours_50000/gt'/image for folder in NAMES.values()]
        if hashlib.sha256(gt[0].read_bytes()).digest()!=hashlib.sha256(gt[1].read_bytes()).digest():
            raise ValueError('Paired GT images differ')
    delta = series['V4']-series['V3']
    rows = [{'test_index':i,'render_file':key,'original_frame_id':int(ids[i]),
             'normalized_time':frames[i]['time'],'V3_psnr_db':float(series['V3'][i]),
             'V4_psnr_db':float(series['V4'][i]),'V4_minus_V3_db':float(delta[i])}
            for i,key in enumerate(expected)]
    bins = np.arange(30,50.001,2)
    summary = {'protocol':'Rubber Duck; time-disjoint120train/30val/50test; final50k snapshot; full-image PNG PSNR',
        'checkpoint':'ours_50000 for both; validation-selected best PSNR also selects iteration50000',
        'model_definition':{'V3':'shared frame clock + V3 image objective; iso/kinematic off',
                            'V4':'V3 + weak velocity/trajectory derivative consistency'},
        'statistics':{k:stats(v) for k,v in series.items()},'paired_delta':stats(delta),
        'V4_higher_frames':int((delta>0).sum()),'V4_lower_frames':int((delta<0).sum()),
        'ties':int((delta==0).sum()),'histogram_edges_db':bins.tolist(),
        'histogram_counts':{k:np.histogram(v,bins)[0].tolist() for k,v in series.items()},
        'lowest_frames':{k:[rows[int(i)] for i in np.argsort(v)[:5]] for k,v in series.items()},
        'largest_V4_gains':[rows[int(i)] for i in np.argsort(delta)[-5:][::-1]],
        'largest_V4_losses':[rows[int(i)] for i in np.argsort(delta)[:5]],
        'validation':['50 unique paired frames','byte-identical GT PNGs','mean matches evaluator within1e-5'],
        'sources':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'limitations':['single scene/seed','correlated frames; distribution is descriptive, not repeated-run evidence',
                       'historical official test exposure persists','full-image PSNR includes background',
                       'population standard deviation across50 frames; quantiles use NumPy linear interpolation']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    with (OUT/'per_frame.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','savefig.facecolor':'white'})
    colors = {'V3':'#277da8','V4':'#dc7a28'}
    fig,axes = plt.subplots(2,2,figsize=(13,8.8),layout='constrained')
    for name,v in series.items():
        axes[0,0].hist(v,bins=bins,histtype='step',lw=2.3,color=colors[name],
                       label=f'{name}: mean {v.mean():.4f} dB')
        axes[0,1].step(np.sort(v),np.arange(1,51)/50,where='post',lw=2,color=colors[name],label=name)
        axes[1,0].plot(ids,v,'o-',ms=3,lw=1.5,color=colors[name],label=name)
    axes[0,0].set(title='A. Per-image PSNR distribution (shared 2-dB bins)',xlabel='PSNR (dB)',ylabel='Number of test images',xticks=bins)
    axes[0,0].legend(fontsize=10)
    axes[0,1].set(title='B. Empirical cumulative distribution',xlabel='PSNR (dB)',ylabel='Fraction of test images',ylim=(0,1.03))
    axes[0,1].legend()
    axes[1,0].set(title='C. Paired PSNR over acquisition order',xlabel='Original frame ID (test every 4 frames)',ylabel='PSNR (dB)')
    axes[1,0].legend()
    axes[1,1].bar(ids,delta,width=2.8,color=np.where(delta>=0,'#2d9276','#c34a56'))
    axes[1,1].axhline(0,color='#333333',lw=.8)
    axes[1,1].axhline(delta.mean(),color='#6d519e',ls='--',lw=1.4,label=f'Mean: {delta.mean():+.4f} dB')
    axes[1,1].set(title=f'D. V4 - V3: {summary["V4_higher_frames"]}/50 images improve',
                   xlabel='Original frame ID',ylabel='Paired PSNR difference (dB)')
    axes[1,1].legend(fontsize=10)
    for ax in axes.flat: ax.grid(alpha=.18,axis='y'); ax.set_axisbelow(True)
    fig.suptitle('V3 vs V4 | same 50 official test images | time-disjoint resplit | final50k',fontsize=14)
    fig.savefig(OUT/'psnr_distribution.png',dpi=180)
    fig.savefig(OUT/'psnr_distribution.svg')
    plt.close(fig)
    lines = ['# V3 / V4 fixed-test PSNR distribution','',summary['protocol'],'',
        'V3: shared frame clock, V3 RGB objective, iso off. V4: V3 plus velocity/trajectory consistency.',
        'Final50k is used for both. Validation-selected PSNR best also equals50k. No new training or test-based selection is performed.','',
        '| Statistic (dB) | V3 | V4 |','|---|---:|---:|']
    labels = {'mean':'Mean','median':'Median','population_std':'Std (population)','min':'Minimum',
              'p10':'10th percentile','q1':'25th percentile','q3':'75th percentile','p90':'90th percentile','max':'Maximum'}
    for key,label in labels.items():
        lines.append(f'| {label} | {summary["statistics"]["V3"][key]:.4f} | {summary["statistics"]["V4"][key]:.4f} |')
    lines += ['',f'Paired V4 minus V3: mean {delta.mean():+.6f} dB; median {np.median(delta):+.6f} dB; '
        f'V4 improves {summary["V4_higher_frames"]}/50 and declines on {summary["V4_lower_frames"]}/50.',
        f'Per-frame difference range: {delta.min():+.6f} to {delta.max():+.6f} dB.',
        'Distributions nearly coincide. V4 shows a small PSNR shift, not a substantial across-frame gain.',
        'Frames are temporally correlated; this single-scene, single-seed distribution does not establish repeated-run statistical significance.',
        'These are full-image PSNR values, including background, not foreground-only scores.',
        'Historical test exposure remains disclosed; the current separation does not undo it.','',
        '![Distribution](psnr_distribution.png)','', '[Per-frame CSV](per_frame.csv) | [Full statistics](summary.json) | [Vector figure](psnr_distribution.svg)']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(OUT),'statistics':summary['statistics'],'paired_delta':summary['paired_delta'],
                      'improved_frames':summary['V4_higher_frames'],'largest_gain':summary['largest_V4_gains'][0],
                      'largest_loss':summary['largest_V4_losses'][0]},indent=2))


if __name__ == '__main__': main()
