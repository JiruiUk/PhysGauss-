"""Compare no-K motion density and direct motion/spatial graph candidates."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import sklearn

from variants.object_centers.audit import ROOT,SOURCE,OUTPUT as EVIDENCE_ROOT,digest,write
from variants.object_centers.motion_review import palette,render_candidates,plots_and_verify,OUTPUT as PREVIOUS_REVIEW
from variants.object_centers.adaptive_grouping import density_motion,spatial_motion_graph

OUTPUT=ROOT/'runs/pidg_object_centers/grouping_review/adaptive_noK_final50000'
REPORTS=ROOT/'experiments/object_centers'
DOC='https://scikit-learn.org/1.8/modules/generated/sklearn.cluster.HDBSCAN.html'


def source_check(summary):
    for path,expected in summary['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('Frozen C1 changed: '+path)
    if digest(EVIDENCE_ROOT/'kernel_evidence.npz')!=summary['evidence_sha256']:
        raise ValueError('Cached evidence changed')


def prepare():
    evidence=EVIDENCE_ROOT/'kernel_evidence.npz';data=np.load(evidence)
    provenance=json.loads((EVIDENCE_ROOT/'source.json').read_text(encoding='utf-8'))
    summary={'status':'preparing','source':'C1_repaired final50000','source_sha256':provenance['source_sha256'],
        'evidence_sha256':digest(evidence),'training_frame_ids':data['frame_ids'].tolist(),
        'fit_split':'train','validation_RGB_loaded':False,'test_RGB_loaded':False,
        'requested_group_count':None,'KMeans_used':False,'new_model_training_started':False,
        'true_instance_labels':False,'sklearn_version':sklearn.__version__,'HDBSCAN_documentation':DOC,
        'density_clustering_weights':'unweighted eligible kernel counts; opacity/vote weights only used for center statistics',
        'feature_rule':'same time-weighted normalized displacement and displacement+trajectory-derived-velocity as preceding review; no RGB or absolute XYZ in density features',
        'sensitivity':'minimum cluster size128 and256, not requested cluster count; HDBSCAN min_samples16/eom/allow_single_cluster=True',
        'graph_rule':'direct reference-pose16NN; normalized motion distance<=0.15; spatial distance<=min(0.08,2.5*min(local16NN radii)) at>=80% of12 sampled training poses; no preliminary grouping',
        'candidates':[],'output_absolute':str(OUTPUT),'gallery_absolute':str(OUTPUT/'index.html')}
    source_check(summary);OUTPUT.mkdir(parents=True,exist_ok=True)
    poses=data['positions'];times=(data['frame_ids']-1)/199;weights=data['weights'];eligible=data['eligible']
    for minimum in (128,256):
        for method,mode in [('N1_HDB_displacement','displacement'),
                            ('N2_HDB_displacement_velocity','displacement_velocity'),
                            ('N3_direct_motion_spatial','displacement_velocity')]:
            start=time.monotonic();name=f'{method}_min{minimum}';folder=OUTPUT/name;folder.mkdir(exist_ok=True)
            print('Clustering '+name+' without requested group count...',flush=True)
            if method.startswith('N3'):
                labels,stats=spatial_motion_graph(poses,times,eligible,weights,min_cluster_size=minimum)
            else:
                labels,stats,confidence=density_motion(poses,times,eligible,weights,mode,min_cluster_size=minimum)
                np.save(folder/'membership_strength.npy',confidence)
            count=int(labels.max()+1);p=palette(count);centers=[];groups=[]
            np.save(folder/'kernel_group_ids.npy',labels)
            for group in range(count):
                mask=labels==group;w=weights[mask];c=np.average(poses[:,mask],axis=1,weights=w)
                centers.append(c);delta=poses[:,mask]-poses[0,mask][None];dc=c-c[0]
                residual=np.sqrt(np.average(np.sum((delta-dc[:,None])**2,axis=-1),axis=1,weights=w).mean())
                groups.append({'id':group,'kernels':int(mask.sum()),'color_rgb':p[group].tolist(),
                    'center_motion_rms':float(np.sqrt(np.sum(dc**2,axis=-1).mean())),
                    'within_group_displacement_rms':float(residual)})
            centers=np.stack(centers,axis=1) if centers else np.zeros((len(times),0,3))
            np.savez_compressed(folder/'center_trajectories.npz',centers=centers,times=times,frame_ids=data['frame_ids'])
            info={'name':name,'method':method,'feature_mode':mode,'requested_group_count':None,
                'min_cluster_size':minimum,'actual_groups':count,'assigned_kernels':int((labels>=0).sum()),
                'unassigned_kernels':int((labels<0).sum()),'total_kernels':len(labels),'groups':groups,
                'algorithm_parameters':stats,'clustering_elapsed_seconds':time.monotonic()-start}
            write(folder/'summary.json',info);summary['candidates'].append(info)
            write(OUTPUT/'summary.json',summary)
            print(f"Prepared {name}: {count} groups, {info['assigned_kernels']} assigned, {info['unassigned_kernels']} unknown ({info['clustering_elapsed_seconds']:.1f}s)",flush=True)
    summary['status']='prepared';write(OUTPUT/'summary.json',summary)
    return summary


def gallery(summary):
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C1：不预设组数的运动聚类</title><style>body{margin:24px;background:#142033;color:#eef5ff;font:16px system-ui}a{color:#9be0b8}select,button{padding:9px;margin:6px}input{width:30%}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #53677d}.note{padding:16px;background:#403a2a}</style>
<h1>C1：不预设组数的运动聚类</h1><p>本轮没有 K、baseK 或 K-means 初始化。最终组数由密度聚类或直接运动/空间图产生；这不等于自动识别出真实物体数。</p>
<p>N1：纯轨迹 HDBSCAN；N2：轨迹＋差分速度 HDBSCAN；N3：直接运动相似＋跨时刻空间邻接图。比较 min_cluster_size=128/256，它是最小簇点数，不是目标组数。灰色表示未分配。</p>
<p>仅读取12个训练时刻；C1权重未改变，尚未训练中心或接触模块。HDBSCAN membership strength不是物体身份准确率或经过校准的概率。</p>
<p class="note" id="finding">候选仍需检查是否存在混组、拆组；不强制把噪声点分配给最近中心。</p>
<p><a href="../motion_methods_final50000/index.html">上一轮固定K比较</a> | <a href="overview_frame0002.png">frame2总览</a> | <a href="overview_frame0146.png">frame146总览</a> | <a href="overview_frame0200.png">frame200总览</a> | <a href="qa.json">校验</a> | <a href="DOC">HDBSCAN参数说明</a></p>
<select id="candidate"></select><button id="prev">上一帧</button><input id="frame" type="range" min="0" max="11" value="0"><button id="next">下一帧</button><span id="position"></span> <a id="png">打开原图</a>
<img id="image"><p id="counts"></p><h2>GT | C1 | N1 | N2 | N3</h2><a id="comparisonLink">打开五列原图</a><img id="comparison">
<h2>已有C1预测的中心轨迹（最多显示24组）</h2><img id="trends"><p><a id="ids">固定kernel ID</a> | <a href="summary.json">完整结果</a></p>
<table><thead><tr><th>组</th><th>kernel数</th><th>中心位移RMS</th><th>组内相对位移RMS</th></tr></thead><tbody id="rows"></tbody></table>
<script>const s=DATA,c=document.getElementById('candidate'),f=document.getElementById('frame');s.candidates.forEach(i=>{let o=document.createElement('option');o.value=i.name;o.textContent=i.name+' / '+i.actual_groups+' groups';c.appendChild(o)});function update(){const i=s.candidates.find(i=>i.name===c.value),n=String(f.value).padStart(5,'0'),p=i.name+'/'+n+'.png',q='comparison_min_cluster_size'+i.min_cluster_size+'/'+n+'.png';document.getElementById('image').src=p;document.getElementById('png').href=p;document.getElementById('comparison').src=q;document.getElementById('comparisonLink').href=q;document.getElementById('trends').src=i.name+'/center_trends.png';document.getElementById('ids').href=i.name+'/kernel_group_ids.npy';document.getElementById('position').textContent='train frame '+s.training_frame_ids[+f.value]+' | '+(+f.value+1)+'/12';document.getElementById('counts').textContent='No requested K | min_cluster_size='+i.min_cluster_size+' | '+i.actual_groups+' groups | '+i.assigned_kernels+'/'+i.total_kernels+' assigned | '+i.unassigned_kernels+' unknown';document.getElementById('rows').innerHTML=i.groups.map(g=>'<tr><td style="color:rgb('+g.color_rgb.map(v=>Math.round(v*255)).join(',')+')">G'+g.id+'</td><td>'+g.kernels+'</td><td>'+g.center_motion_rms.toFixed(4)+'</td><td>'+g.within_group_displacement_rms.toFixed(4)+'</td></tr>').join('');if(s.visual_review)document.getElementById('finding').textContent=s.visual_review;}function step(n){f.value=Math.max(0,Math.min(11,+f.value+n));update()}c.onchange=f.oninput=update;document.getElementById('prev').onclick=()=>step(-1);document.getElementById('next').onclick=()=>step(1);update();</script></html>'''
    (OUTPUT/'index.html').write_text(page.replace('DOC',DOC).replace('DATA',json.dumps(summary,ensure_ascii=False)),encoding='utf-8')
    for parent in (PREVIOUS_REVIEW,EVIDENCE_ROOT):
        path=parent/'index.html'
        if path.exists():
            old=path.read_text(encoding='utf-8')
            if 'id="adaptive-noK-link"' not in old:
                link='<p id="adaptive-noK-link"><a href="../adaptive_noK_final50000/index.html">新增：不预设组数的聚类结果</a></p>'
                path.write_text(old.replace('</h1>','</h1>'+link,1),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--render-existing',action='store_true');args=parser.parse_args()
    summary=json.loads((OUTPUT/'summary.json').read_text(encoding='utf-8')) if args.render_existing else prepare()
    source_check(summary)
    if args.prepare_only:return
    render_candidates(summary,OUTPUT,'min_cluster_size')
    plots_and_verify(summary,OUTPUT,'min_cluster_size',
        extra_code_files=[Path(__file__),ROOT/'variants/object_centers/adaptive_grouping.py',ROOT/'tests/test_adaptive_grouping.py'])
    summary['status']='completed';write(OUTPUT/'summary.json',summary)
    write(REPORTS/'adaptive_grouping_review.json',summary);gallery(summary)
    print('Completed no-K gallery: '+str(OUTPUT/'index.html'),flush=True)


if __name__=='__main__':main()
