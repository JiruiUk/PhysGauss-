"""English figure annotations; reuse rendered pixels and cached diagnostic data."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
TMP=ROOT/'tmp/pdfs/weekly_en_20261103'
SNAP=ROOT/'tmp/pdfs/weekly_20261103/evidence'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
    out=TMP/'figures';out.mkdir(parents=True,exist_ok=True)
    evidence=TMP/'figure_evidence';evidence.mkdir(parents=True,exist_ok=True)
    sources={}
    def snapshot(rel):
        src=ROOT/rel;dst=evidence/rel
        if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
        sources[str(rel)]={'sha256':sha(dst),'snapshot':str(dst.relative_to(ROOT))}
        return dst
    original=SNAP/'output/fixed_camera_temporal/key_comparison.png'
    im=Image.open(original).convert('RGB');before=np.array(im);draw=ImageDraw.Draw(im)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    labels=['Observed GT (original path)','C1 render (original path)','C1 render (fixed camera86)','Fixed-camera alpha (white=opaque)']
    mask=np.zeros((im.height,im.width),dtype=bool)
    for row,label in enumerate(labels):
        y=row*252
        draw.rectangle((0,y,1199,y+26),fill='#181b22');mask[y:y+27]=True
        for col,t in enumerate([2,86,200]):draw.text((col*400+8,y+5),f'{label} | time {t}',font=font,fill='white')
    after=np.array(im)
    assert np.array_equal(before[~mask],after[~mask])
    im.save(out/'key_comparison_en.png')
    sources[str(original.relative_to(ROOT))]={'sha256':sha(original),'operation':'English labels replace only 27-pixel annotation strips; all rendered pixels outside strips verified identical'}
    raw='runs/pidg_frozen_velocity_probe/H3_final_20261101'
    result=json.loads(snapshot(Path('experiments/frozen_velocity_probe/results.json')).read_text(encoding='utf-8'))
    control=json.loads(snapshot(Path('experiments/frozen_velocity_probe/clock_control_result.json')).read_text(encoding='utf-8'))
    arms=result['arms']+[control]
    detail=json.loads(snapshot(Path('experiments/frozen_velocity_probe/temporal_details.json')).read_text(encoding='utf-8'))
    names=[arm['arm']['name'] for arm in arms]
    histories={name:[json.loads(s) for s in snapshot(Path(raw)/name/'history.jsonl').read_text().splitlines()] for name in names}
    for arm in arms:
        name=arm['arm']['name']
        for split,v in histories[name][-1]['metrics'].items():
            assert abs(v['relative_rmse']-arm['final']['metrics'][split]['relative_rmse'])<1e-12
    os.environ.setdefault('MPLCONFIGDIR',str(TMP/'mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.unicode_minus':False})
    palette=['#1b678d','#d47d30','#389777','#9165a5']
    labels={'B1_pretrained_lr1e-3':'B1: pretrained / 1e-3','B2_pretrained_lr3e-4':'B2: pretrained / 3e-4',
        'B3_scratch_lr1e-3':'B3: random init / 1e-3','C1_fixed_monotonic_lr1e-3':'C: fixed monotonic clock'}
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained')
    a=histories['A_small_overfit']
    axes[0,0].plot([r['step'] for r in a],[r['metrics']['fit']['relative_rmse']*100 for r in a],color=palette[0],marker='o')
    axes[0,0].set(title='Small-sample fit: 128 points x 8 times',ylabel='Relative RMSE (%)',xlabel='Optimization step',yscale='log')
    for j,name in enumerate(names[1:]):
        h=histories[name]
        for ax,key in [(axes[0,1],'known_train'),(axes[1,0],'known_validation')]:
            ax.plot([r['step'] for r in h],[r['metrics'][key]['relative_rmse']*100 for r in h],label=labels[name],color=palette[j])
    axes[0,1].set(title='Broad fit: 2048 points x 120 times',ylabel='Training relative RMSE (%)',xlabel='Optimization step')
    axes[1,0].set(title='Time holdout: same points x 30 unseen times',ylabel='Time-holdout relative RMSE (%)',xlabel='Optimization step')
    for name,values in detail['latent'].items():
        if name=='A_small_overfit':continue
        axes[1,1].plot(np.linspace(0,1,len(values['values'])),values['values'],
            label='Original model' if name=='source' else labels[name],
            color='#53616d' if name=='source' else palette[names[1:].index(name)],linestyle='--' if name=='source' else '-')
    axes[1,1].set(title='Internal scalar time of velocity network',ylabel='Internal time q(t)',xlabel='Shared normalized time')
    for ax in axes.flat:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    for ax in [axes[0,1],axes[1,0],axes[1,1]]:ax.legend(fontsize=7.7)
    fig.suptitle('Frozen H3 trajectories: velocity fitting and internal clocks',fontsize=14,fontweight='bold')
    fig.savefig(out/'velocity_fit_en.png',dpi=190);plt.close(fig)
    record={'sources':sources,'same_render_pixels_outside_labels':True,'history_final_values_match_records':True,
        'new_model_inference':False,'operations':['translate existing annotation strips','replot saved histories and cached latent arrays with English labels'],
        'derived_images':{p.name:sha(p) for p in out.glob('*.png')}}
    (TMP/'english_figure_sources.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'output/fixed_camera_temporal/key_comparison.png':str(out/'key_comparison_en.png'),
        'atlas/velocity.png':str(out/'velocity_fit_en.png')},record

if __name__=='__main__':prepare()
