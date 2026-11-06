"""Fixed-layout descriptive figures from every preregistered render."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
REPORT=BASE/'v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_view_time_audit/v1'

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def panel(paths,labels,title,destination,columns,highlight=None,size=(240,135)):
    rows=(len(paths)+columns-1)//columns;w,h=size
    canvas=Image.new('RGB',(columns*w,60+rows*(h+27)),(30,30,30));draw=ImageDraw.Draw(canvas)
    draw.text((8,8),title,fill='white')
    for n,(path,label) in enumerate(zip(paths,labels)):
        x=(n%columns)*w;y=60+(n//columns)*(h+27)
        with Image.open(path) as im:canvas.paste(im.convert('RGB').resize(size,Image.Resampling.LANCZOS),(x,y))
        draw.text((x+3,y+h+3),label,fill='white')
        if highlight and highlight[n]:draw.rectangle((x,y,x+w-1,y+h-1),outline=(0,255,160),width=3)
    destination.parent.mkdir(parents=True,exist_ok=True);canvas.save(destination)

def main():
    p=read(BASE/'protocol.json');all_data={name:read(REPORT/(name+'.json')) for name in p['models']}
    summary={};records=[]
    for name,data in all_data.items():
        rows={(r['camera_index'],r['time_index']):r for r in data['rows']}
        summary[name]={}
        for anchor in p['anchors']:
            ids=[anchor+d for d in p['offsets']]
            for label,suffix in [('joint','joint'),('dynamic alpha','alpha')]:
                paths=[OUT/name/'cells'/f'c{c:03d}_t{t:03d}_{suffix}.png' for t in ids for c in ids]
                labels=[f'camera {c} | time index {t}' for t in ids for c in ids]
                panel(paths,labels,f'{name}, anchor {anchor}: {label}. Rows=time; columns=camera. Green=observed pair.',
                    OUT/'figures'/f'{name}_a{anchor}_{suffix}_cross.png',7,[c==t for t in ids for c in ids])
            paths=[OUT/name/'cells'/f'c{i:03d}_t{i:03d}_{kind}.png' for kind in ['observed','joint'] for i in ids]
            panel(paths,[f'{kind} pair {i}' for kind in ['observed','joint'] for i in ids],
                f'{name}, anchor {anchor}. Top=actual RGB observations; bottom=matched native model render.',
                OUT/'figures'/f'{name}_a{anchor}_observed.png',7)
            for c in p['all_camera_indices']:
                r=rows[c,anchor];d=r['dynamic'];shape=d['shape_proxies'];dist=r['camera_time_distance']
                records.append(dict(model=name,anchor=anchor,camera=c,matched_camera=c==anchor,**dist,
                    alpha_gt05_pixels=d['alpha_gt05_pixels'],alpha_mass=d['alpha_mass'],
                    dark_high_alpha_pixels=d['dark_high_alpha_pixels'],dark_fraction=d['dark_fraction_on_alpha'],
                    components=shape['component_count'],thin_support=shape['thin_support_fraction'],
                    small_islands=shape['small_island_fraction'],holes=shape['enclosed_hole_fraction'],clipped=shape['touches_image_border']))
            summary[name][str(anchor)]=dict(observed_PSNR=rows[anchor,anchor]['observed_PSNR'],
                fixed_time_local=[dict(offset=c-anchor,camera=c,**rows[c,anchor]['camera_time_distance'],**rows[c,anchor]['dynamic']) for c in ids],
                same_camera_time=[dict(offset=t-anchor,time_index=t,time=rows[anchor,t]['time'],**rows[anchor,t]['dynamic']) for t in ids])
    for anchor in p['anchors']:
        ids=[anchor+d for d in p['offsets']]
        for suffix,title in [('joint','Joint RGB'),('dynamic','Dynamic RGB without static occlusion'),('gray','Constant gray dynamic support'),('alpha','Dynamic alpha')]:
            paths=[OUT/name/'native1600'/f'c{c:03d}_t{anchor:03d}_{suffix}.png' for name in p['models'] for c in ids]
            labels=[f'{name} | camera {c} | d={c-anchor:+}' for name in p['models'] for c in ids]
            panel(paths,labels,f'{title}, time index {anchor}; rows=J7/control/memory. Green=matched camera.',
                OUT/'figures'/f'a{anchor}_{suffix}_comparison.png',7,[c==anchor for name in p['models'] for c in ids])
    with (REPORT/'fixed_time_all_views.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (REPORT/'paired_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    colors=['#1777bb','#d08012','#189650']
    fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    for row,anchor in enumerate(p['anchors']):
        for name,color in zip(p['models'],colors):
            q=[r for r in records if r['model']==name and r['anchor']==anchor]
            for col,(key,label) in enumerate([('dark_fraction','Dark fraction within alpha > .5'),('thin_support','Thin support fraction (proxy)'),('alpha_gt05_pixels','Alpha > .5 area (pixels)')]):
                ax=axes[row,col];ax.scatter([r['view_direction_deg'] for r in q],[r[key] for r in q],s=8,alpha=.45,color=color,label=name)
                matched=next(r for r in q if r['matched_camera']);ax.scatter([matched['view_direction_deg']],[matched[key]],s=65,marker='*',color=color)
                ax.set(xlabel='Angle from observed camera (deg)',ylabel=label,title=f'Time index {anchor}');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('All 120 camera poses per fixed time. Descriptive proxies, no novel-view GT.',fontsize=13)
    fig.savefig(OUT/'figures'/'angle_sweep.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
    for row,anchor in enumerate(p['anchors']):
        for name,color in zip(p['models'],colors):
            q=summary[name][str(anchor)]
            for col,field in enumerate(['dark_fraction_on_alpha','alpha_gt05_pixels']):
                for key,style in [('fixed_time_local','-'),('same_camera_time','--')]:
                    axes[row,col].plot([r['offset'] for r in q[key]],[r[field] for r in q[key]],style,color=color,label=f'{name} {key}')
                axes[row,col].set(title=f'Anchor {anchor}: {field}',xlabel='Train-list index offset');axes[row,col].grid(alpha=.2)
    axes[0,0].legend(fontsize=6);fig.suptitle('Solid: change camera at fixed time. Dashed: change time at fixed camera. No GT.')
    fig.savefig(OUT/'figures'/'paired_controls.png',dpi=150);plt.close(fig)
    print('V0_REPORT_FIGURES_COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
