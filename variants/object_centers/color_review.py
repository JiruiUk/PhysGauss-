"""No-K motion-plus-color audit; compare against frozen motion-only candidates."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from variants.object_centers.audit import ROOT,OUTPUT as EVIDENCE_ROOT,write,digest
from variants.object_centers.adaptive_review import OUTPUT as BASELINE,source_check
from variants.object_centers.adaptive_grouping import density_motion,spatial_motion_graph
from variants.object_centers.motion_review import palette,render_candidates,plots_and_verify,OUTPUT as MOTION_REVIEW

OUTPUT=ROOT/'runs/pidg_object_centers/grouping_review/motion_color_noK_final50000'
REPORTS=ROOT/'experiments/object_centers'
COLOR_WEIGHT=1.
COLOR_RADIUS=.35


def prepare():
    previous=json.loads((BASELINE/'summary.json').read_text(encoding='utf-8'));source_check(previous)
    data=np.load(EVIDENCE_ROOT/'kernel_evidence.npz');poses=data['positions'];times=(data['frame_ids']-1)/199
    eligible=data['eligible'];weights=data['weights'];colors=data['mean_colors']
    # One real full-resolution-kernel control, in addition to unit flags-off tests.
    print('Verify color-disabled density and graph against saved no-color labels...',flush=True)
    control,_,confidence=density_motion(poses,times,eligible,weights,'displacement',colors=colors,color_weight=0.)
    assert np.array_equal(control,np.load(BASELINE/'N1_HDB_displacement_min128/kernel_group_ids.npy'))
    assert np.array_equal(confidence,np.load(BASELINE/'N1_HDB_displacement_min128/membership_strength.npy'))
    control,_=spatial_motion_graph(poses,times,eligible,weights,colors=colors,color_radius=None)
    assert np.array_equal(control,np.load(BASELINE/'N3_direct_motion_spatial_min128/kernel_group_ids.npy'))
    summary={'status':'preparing','source':previous['source'],'source_sha256':previous['source_sha256'],
        'evidence_sha256':previous['evidence_sha256'],'training_frame_ids':previous['training_frame_ids'],
        'fit_split':'train','validation_RGB_loaded':False,'test_RGB_loaded':False,'GT_RGB_used_as_color_feature':False,
        'requested_group_count':None,'KMeans_used':False,'new_model_training_started':False,'true_instance_labels':False,
        'color_feature':'cached mean of C1 positive-SH RGB (clipped0..1) at12 training view/time pairs; includes invisible kernel/view pairs; not intrinsic albedo',
        'density_color_rule':'append color_weight*mean_RGB to original normalized motion features',
        'density_color_weight':COLOR_WEIGHT,'graph_color_radius_RGB':COLOR_RADIUS,
        'color_disabled_real_control_density_labels_and_membership_exact_match':True,
        'color_disabled_real_control_graph_labels_exact_match':True,
        'baseline_summary_sha256':digest(BASELINE/'summary.json'),
        'density_weights':'same unweighted eligible kernel counts; opacity/vote weights only for centers',
        'candidates':[],'output_absolute':str(OUTPUT),'gallery_absolute':str(OUTPUT/'index.html')}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    methods=[('R1_motion_RGB','N1_HDB_displacement','displacement'),
             ('R2_motion_velocity_RGB','N2_HDB_displacement_velocity','displacement_velocity'),
             ('R3_motion_spatial_RGB','N3_direct_motion_spatial','displacement_velocity')]
    for minimum in (128,256):
        for method,baseline,mode in methods:
            start=time.monotonic();name=f'{method}_min{minimum}';folder=OUTPUT/name;folder.mkdir(exist_ok=True)
            print('Clustering '+name+' (no requested group count)...',flush=True)
            if method.startswith('R3'):
                labels,params=spatial_motion_graph(poses,times,eligible,weights,min_cluster_size=minimum,
                                                  colors=colors,color_radius=COLOR_RADIUS)
            else:
                labels,params,confidence=density_motion(poses,times,eligible,weights,mode,min_cluster_size=minimum,
                                                        colors=colors,color_weight=COLOR_WEIGHT)
                np.save(folder/'membership_strength.npy',confidence)
            count=int(labels.max()+1);p=palette(count);groups=[];centers=[]
            np.save(folder/'kernel_group_ids.npy',labels)
            for g in range(count):
                mask=labels==g;w=weights[mask];c=np.average(poses[:,mask],axis=1,weights=w);centers.append(c)
                delta=poses[:,mask]-poses[0,mask][None];dc=c-c[0]
                error=np.sqrt(np.average(np.sum((delta-dc[:,None])**2,axis=-1),axis=1,weights=w).mean())
                groups.append({'id':g,'kernels':int(mask.sum()),'color_rgb':p[g].tolist(),
                    'center_motion_rms':float(np.sqrt(np.sum(dc**2,axis=-1).mean())),
                    'within_group_displacement_rms':float(error),'mean_C1_RGB':np.average(colors[mask],axis=0,weights=w).tolist()})
            centers=np.stack(centers,axis=1) if centers else np.zeros((len(times),0,3))
            np.savez_compressed(folder/'center_trajectories.npz',centers=centers,times=times,frame_ids=data['frame_ids'])
            baseline_name=f'{baseline}_min{minimum}'
            old=next(i for i in previous['candidates'] if i['name']==baseline_name)
            info={'name':name,'method':method,'feature_mode':mode,'requested_group_count':None,'min_cluster_size':minimum,
                'actual_groups':count,'assigned_kernels':int((labels>=0).sum()),'unassigned_kernels':int((labels<0).sum()),
                'total_kernels':len(labels),'groups':groups,'algorithm_parameters':params,
                'baseline_name':baseline_name,'baseline_groups':old['actual_groups'],
                'baseline_assigned_kernels':old['assigned_kernels'],
                'baseline_labels_sha256':digest(BASELINE/baseline_name/'kernel_group_ids.npy'),
                'clustering_elapsed_seconds':time.monotonic()-start}
            write(folder/'summary.json',info);summary['candidates'].append(info);write(OUTPUT/'summary.json',summary)
            print(f"{name}: {old['actual_groups']} -> {count} groups; {info['assigned_kernels']} assigned ({info['clustering_elapsed_seconds']:.1f}s)",flush=True)
    summary['status']='prepared';write(OUTPUT/'summary.json',summary)
    return summary


def before_after(summary):
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21);checked=0
    for info in summary['candidates']:
        folder=OUTPUT/'before_after'/info['name'];folder.mkdir(parents=True,exist_ok=True)
        if digest(BASELINE/info['baseline_name']/'kernel_group_ids.npy')!=info['baseline_labels_sha256']:
            raise ValueError('Previous candidate labels changed')
        for index,frame in enumerate(summary['training_frame_ids']):
            with Image.open(BASELINE/info['baseline_name']/f'{index:05d}.png') as old, \
                 Image.open(OUTPUT/info['name']/f'{index:05d}.png') as new:
                assert np.array_equal(np.asarray(old)[50:500,:1600],np.asarray(new)[50:500,:1600])
                panels=[old.crop((0,50,800,500)),old.crop((800,50,1600,500)),
                        old.crop((1600,50,2400,500)),new.crop((1600,50,2400,500))]
            canvas=Image.new('RGB',(3200,500),'#edf2f7');d=ImageDraw.Draw(canvas)
            titles=[f'GT / train frame{frame:04d}','Frozen C1 render',
                    f"Motion only / {info['baseline_groups']} groups",f"+ RGB / {info['actual_groups']} groups"]
            for j,(title,panel) in enumerate(zip(titles,panels)):
                d.text((j*800+10,12),title,font=font,fill='#17304a');canvas.paste(panel,(j*800,50))
            path=folder/f'{index:05d}.png';canvas.save(path)
            with Image.open(path) as im:
                for j,panel in enumerate(panels):assert np.array_equal(np.asarray(im)[50:500,j*800:(j+1)*800],np.asarray(panel))
            checked+=1
    return checked


def gallery(summary):
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C1：运动轨迹＋颜色（不设组数）</title><style>body{margin:24px;background:#142033;color:#eef5ff;font:16px system-ui}a{color:#9be0b8}select,button{padding:9px;margin:6px}input{width:30%}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #53677d}.note{padding:16px;background:#403a2a}</style>
<h1>C1：运动轨迹＋颜色，不预设组数</h1><p>R1：纯轨迹＋RGB HDBSCAN；R2：轨迹＋速度＋RGB HDBSCAN；R3：运动/空间图＋RGB相似边。没有K或K-means初始化；min128/256为最小簇规模。</p>
<p>颜色来自C1预测SH在12个训练视角/时刻的RGB均值，包含不可见kernel/view对；不使用GT颜色，也不是物理材质色。前两种追加1.0×RGB特征，第三种要求边的RGB距离≤0.35。灰色表示未知。</p>
<p class="note" id="finding">不同候选的组编号和调色板不对应同一个物体；检查物体是否合组/拆组，而非比较某种颜色本身。没有新增模型训练。</p>
<p><a href="../adaptive_noK_final50000/index.html">无颜色对照</a> | <a href="overview_frame0002.png">frame2总览</a> | <a href="overview_frame0146.png">frame146总览</a> | <a href="overview_frame0200.png">frame200总览</a> | <a href="qa.json">校验</a></p>
<select id="candidate"></select><button id="prev">上一帧</button><input id="frame" type="range" min="0" max="11" value="0"><button id="next">下一帧</button><span id="position"></span><a id="png">打开原图</a>
<img id="image"><p id="counts"></p><h2>GT | C1 | 不加颜色 | 加颜色</h2><a id="pairlink">打开四列原图</a><img id="pair">
<h2>GT | C1 | R1 | R2 | R3</h2><a id="comparisonLink">打开五列原图</a><img id="comparison">
<h2>已有C1中心轨迹（最多24组）</h2><img id="trends"><p><a id="ids">固定kernel ID</a> | <a href="summary.json">完整结果</a></p>
<table><thead><tr><th>组</th><th>kernel数</th><th>中心位移RMS</th><th>组内相对位移RMS</th></tr></thead><tbody id="rows"></tbody></table>
<script>const s=DATA,c=document.getElementById('candidate'),f=document.getElementById('frame');s.candidates.forEach(i=>{let o=document.createElement('option');o.value=i.name;o.textContent=i.name+' / '+i.actual_groups+' groups';c.appendChild(o)});function update(){const i=s.candidates.find(i=>i.name===c.value),n=String(f.value).padStart(5,'0'),p=i.name+'/'+n+'.png',q='comparison_min_cluster_size'+i.min_cluster_size+'/'+n+'.png',b='before_after/'+i.name+'/'+n+'.png';document.getElementById('image').src=p;document.getElementById('png').href=p;document.getElementById('pair').src=b;document.getElementById('pairlink').href=b;document.getElementById('comparison').src=q;document.getElementById('comparisonLink').href=q;document.getElementById('trends').src=i.name+'/center_trends.png';document.getElementById('ids').href=i.name+'/kernel_group_ids.npy';document.getElementById('position').textContent='train frame '+s.training_frame_ids[+f.value]+' | '+(+f.value+1)+'/12';document.getElementById('counts').textContent='No requested K | '+i.baseline_groups+' -> '+i.actual_groups+' groups | assigned '+i.baseline_assigned_kernels+' -> '+i.assigned_kernels+' | unknown '+i.unassigned_kernels;document.getElementById('rows').innerHTML=i.groups.map(g=>'<tr><td style="color:rgb('+g.color_rgb.map(v=>Math.round(v*255)).join(',')+')">G'+g.id+'</td><td>'+g.kernels+'</td><td>'+g.center_motion_rms.toFixed(4)+'</td><td>'+g.within_group_displacement_rms.toFixed(4)+'</td></tr>').join('');if(s.visual_review)document.getElementById('finding').textContent=s.visual_review;}function step(n){f.value=Math.max(0,Math.min(11,+f.value+n));update()}c.onchange=f.oninput=update;document.getElementById('prev').onclick=()=>step(-1);document.getElementById('next').onclick=()=>step(1);update();</script></html>'''
    (OUTPUT/'index.html').write_text(html.replace('DATA',json.dumps(summary,ensure_ascii=False)),encoding='utf-8')
    for parent in (BASELINE,MOTION_REVIEW,EVIDENCE_ROOT):
        path=parent/'index.html'
        if path.exists():
            old=path.read_text(encoding='utf-8')
            if 'id="motion-color-link"' not in old:
                link='<p id="motion-color-link"><a href="../motion_color_noK_final50000/index.html">新增：运动轨迹＋颜色聚类</a></p>'
                path.write_text(old.replace('</h1>','</h1>'+link,1),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render-existing',action='store_true');args=parser.parse_args()
    summary=json.loads((OUTPUT/'summary.json').read_text(encoding='utf-8')) if args.render_existing else prepare()
    source_check(summary)
    render_candidates(summary,OUTPUT,'min_cluster_size')
    plots_and_verify(summary,OUTPUT,'min_cluster_size',extra_code_files=[Path(__file__),
        ROOT/'variants/object_centers/adaptive_grouping.py',ROOT/'variants/object_centers/color_features.py',
        ROOT/'tests/test_motion_color_grouping.py'])
    checked=before_after(summary);summary.update(status='completed',before_after_PNGs_verified=checked)
    write(OUTPUT/'summary.json',summary);write(REPORTS/'motion_color_review.json',summary)
    qa=json.loads((OUTPUT/'qa.json').read_text(encoding='utf-8'));qa['before_after_PNGs_verified']=checked;write(OUTPUT/'qa.json',qa)
    gallery(summary);print('Completed: '+str(OUTPUT/'index.html'),flush=True)


if __name__=='__main__':main()
