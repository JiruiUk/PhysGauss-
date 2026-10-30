"""Train-only, read-only comparison of three fixed motion grouping methods."""
import argparse
import colorsys
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np

from variants.object_centers.audit import ROOT,SOURCE,OUTPUT as PREVIOUS,PALETTE,digest,write,tensor_image
from variants.object_centers.motion_grouping import cluster_motion,persistent_components

OUTPUT=ROOT/'runs/pidg_object_centers/grouping_review/motion_methods_final50000'
REPORTS=ROOT/'experiments/object_centers'


def palette(size):
    import matplotlib.colors as mc
    return np.array([mc.to_rgb(PALETTE[i]) if i<len(PALETTE) else
                     colorsys.hsv_to_rgb((i*.61803398875)%1,.64,.95) for i in range(size)],dtype=np.float32).reshape(size,3)


def prepare():
    evidence=PREVIOUS/'kernel_evidence.npz'
    provenance=json.loads((PREVIOUS/'source.json').read_text(encoding='utf-8'))
    for path,expected in provenance['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('C1 source changed: '+path)
    loader=PREVIOUS/'train_only_loader'
    if json.loads((loader/'transforms_test.json').read_text(encoding='utf-8'))['frames']:
        raise ValueError('Review loader must have empty test frames')
    data=np.load(evidence)
    poses=data['positions'];times=(data['frame_ids']-1)/199
    eligible=data['eligible'];weights=data['weights'];candidates=[]
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for k in (5,6):
        for method,mode in [('T1_displacement','displacement'),('T2_displacement_velocity','displacement_velocity'),
                            ('T3_motion_connectivity','displacement_velocity')]:
            name=f'{method}_baseK{k}';folder=OUTPUT/name;folder.mkdir(exist_ok=True)
            labels=cluster_motion(poses,times,eligible,weights,k,mode)
            graph=None
            if method.startswith('T3'):
                labels,graph=persistent_components(poses,labels,weights)
            np.save(folder/'kernel_group_ids.npy',labels)
            count=int(labels.max()+1);colors=palette(count);centers=[];groups=[]
            for g in range(count):
                mask=labels==g;w=weights[mask]
                c=np.average(poses[:,mask],axis=1,weights=w);centers.append(c)
                delta=poses[:,mask]-poses[0,mask][None];center_delta=c-c[0]
                error=np.sqrt(np.average(np.sum((delta-center_delta[:,None])**2,axis=-1),axis=1,weights=w).mean())
                groups.append({'id':g,'kernels':int(mask.sum()),'color_rgb':colors[g].tolist(),
                    'center_motion_rms':float(np.sqrt(np.sum(center_delta**2,axis=-1).mean())),
                    'within_group_displacement_rms':float(error)})
            centers=np.stack(centers,axis=1) if centers else np.zeros((len(times),0,3))
            np.savez_compressed(folder/'center_trajectories.npz',centers=centers,times=times,frame_ids=data['frame_ids'])
            info={'name':name,'method':method,'feature_mode':mode,'base_K':k,'actual_groups':count,
                  'assigned_kernels':int((labels>=0).sum()),'unassigned_kernels':int((labels<0).sum()),
                  'total_kernels':len(labels),'groups':groups,'connectivity':graph}
            write(folder/'summary.json',info);candidates.append(info)
            print(f"Prepared {name}: {count} groups, {(labels>=0).sum()} assigned, {(labels<0).sum()} unknown",flush=True)
    summary={'status':'prepared','source':'C1_repaired final50000','source_sha256':provenance['source_sha256'],
        'evidence_sha256':digest(evidence),'training_frame_ids':data['frame_ids'].tolist(),
        'fit_split':'train','validation_RGB_loaded':False,'test_RGB_loaded':False,
        'absolute_position_or_RGB_in_motion_features':False,'velocity_source':'finite differences of deformed kernel trajectories with actual dt',
        'weights':'same fixed opacity * training foreground-candidate vote weights as previous review',
        'displacement_normalization':'single shared 95th percentile trajectory RMS; trapezoidal time weights; no per-kernel amplitude normalization',
        'velocity_normalization':'single shared 95th percentile speed RMS, independent block scale; interval time weights',
        'spatial_rule':'T3 only: same T2 motion group + reference-pose16NN edges, distance <= min(0.08,2.5*min(local16NN radius)), at >=80% of12 sampled times; components<128 kernels become unknown',
        'true_instance_labels':False,'new_model_training_started':False,'candidates':candidates,
        'output_absolute':str(OUTPUT),'gallery_absolute':str(OUTPUT/'index.html')}
    write(OUTPUT/'summary.json',summary)
    return summary


def render_candidates(summary):
    import torch
    from PIL import Image,ImageDraw,ImageFont
    repo=ROOT/'variants/color_repair/code';sys.path.insert(0,str(repo));sys.path.insert(0,str(ROOT))
    os.environ['PIDG_METRIC_CONFIG']=str(SOURCE/'metric_config.json')
    os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
    os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
    from variants.color_repair.run import assert_frozen
    assert_frozen(repo)
    from arguments import ModelParams,PipelineParams
    from scene_PIDG import Scene,DeformModel
    from gaussian_renderer import GaussianModel,render
    parser=argparse.ArgumentParser();model=ModelParams(parser);pipeline=PipelineParams(parser);args=parser.parse_args([])
    for key,value in runpy.run_path(str(SOURCE/'model_config.py')).items():
        if not key.startswith('__'):setattr(args,key,value)
    args.source_path=str(PREVIOUS/'train_only_loader');args.model_path=str(SOURCE);args.resolution=800
    args.load2gpu_on_the_fly=True;args.data_device='cuda:0';args.eval=True
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    evidence=np.load(PREVIOUS/'kernel_evidence.npz')
    with torch.no_grad():
        gaussians=GaussianModel(args.sh_degree)
        scene=Scene(model.extract(args),gaussians,load_iteration='50000',shuffle=False)
        assert len(scene.getTestCameras())==0
        views=sorted(scene.getTrainCameras(),key=lambda v:int(v.image_name))
        assert [int(v.image_name) for v in views]==summary['training_frame_ids']
        if not np.array_equal(gaussians.get_xyz.cpu().numpy(),evidence['canonical']):
            raise ValueError('Cached evidence kernel ordering changed')
        deform=DeformModel(args.grid_args,args.network_args,scale_xyz=args.scale_xyz,reg_spatial_able=False,reg_temporal_able=False)
        deform.load_weights(str(SOURCE),iteration='50000')
        background=torch.zeros(3,device='cuda');runtime=[]
        for info in summary['candidates']:
            folder=OUTPUT/info['name'];labels=np.load(folder/'kernel_group_ids.npy');p=palette(info['actual_groups'])
            rgb=np.tile(np.array([.35,.4,.46],dtype=np.float32),(len(labels),1))
            rgb[labels>=0]=p[labels[labels>=0]]
            runtime.append((info,torch.from_numpy(rgb).cuda(),np.load(folder/'center_trajectories.npz')['centers']))
        max_pose_error=0.
        for index,view in enumerate(views):
            view.load2device();view.fid=view.fid.new_tensor([(int(view.image_name)-1)/199])
            result=deform.step(gaussians.get_xyz.detach(),view.fid.unsqueeze(0).expand(len(gaussians.get_xyz),-1),fixed_attention=True)
            original=render(view,gaussians,pipeline.extract(args),background,result['d_xyz'],result['d_rotation'],result['d_scaling'])
            error=float(np.abs(original['means3D_deform'].cpu().numpy()-evidence['positions'][index]).max())
            max_pose_error=max(error,max_pose_error)
            if error>1e-5:raise ValueError('Cached trajectory mismatch')
            with Image.open(PREVIOUS/'canonical_K5'/f'{index:05d}.png') as old:
                gt=old.crop((0,50,800,500));old_rgb=old.crop((800,50,1600,500))
            rgb=tensor_image(original['render'])
            if not np.array_equal(np.asarray(old_rgb),np.asarray(rgb)):raise ValueError('Frozen C1 RGB mismatch')
            colored={}
            for info,override,centers in runtime:
                package=render(view,gaussians,pipeline.extract(args),background,result['d_xyz'],result['d_rotation'],result['d_scaling'],override_color=override)
                image=tensor_image(package['render']);d=ImageDraw.Draw(image);k=info['actual_groups']
                if k:
                    c=torch.from_numpy(centers[index]).cuda();h=torch.cat((c,torch.ones(k,1,device='cuda')),-1)@view.full_proj_transform
                    pixels=(((h[:,:2]/h[:,3:4])+1)*c.new_tensor([800,450])/2-.5).cpu().numpy()
                    # Limit marker clutter. The full center bank is exported for all groups.
                    visible_ids=sorted(info['groups'],key=lambda g:g['kernels'],reverse=True)[:12]
                    for group in visible_ids:
                        g=group['id'];x,y=pixels[g]
                        if h[g,3]>0 and 0<=x<800 and 0<=y<450:
                            x,y=int(x),int(y);color=tuple(round(v*255) for v in group['color_rgb'])
                            d.ellipse((x-5,y-5,x+5,y+5),fill=color,outline='white',width=1)
                            d.text((x+7,y-10),f'G{g}',font=small,fill='white',stroke_width=2,stroke_fill='black')
                colored[info['name']]=image
                canvas=Image.new('RGB',(2400,545),'#edf2f7');draw=ImageDraw.Draw(canvas)
                for j,(title,panel) in enumerate(zip(['GT / train frame'+view.image_name,'Frozen C1 final50000',info['name']],(gt,rgb,image))):
                    draw.text((j*800+10,12),title,font=font,fill='#17304a');canvas.paste(panel,(j*800,50))
                draw.text((12,510),f"{k} groups | gray: unknown | labels fixed over time | center markers: up to12 largest groups",font=small,fill='#17304a')
                canvas.save(OUTPUT/info['name']/f'{index:05d}.png')
            for k in (5,6):
                items=[i for i in summary['candidates'] if i['base_K']==k]
                folder=OUTPUT/f'comparison_baseK{k}';folder.mkdir(exist_ok=True)
                canvas=Image.new('RGB',(4000,500),'#edf2f7');d=ImageDraw.Draw(canvas)
                for j,(title,panel) in enumerate(zip(['GT / train frame'+view.image_name,'C1 render']+
                    [i['method']+f" / {i['actual_groups']} groups" for i in items],
                    [gt,rgb]+[colored[i['name']] for i in items])):
                    d.text((j*800+10,12),title,font=font,fill='#17304a');canvas.paste(panel,(j*800,50))
                canvas.save(folder/f'{index:05d}.png')
            view.load2device('cpu');print('Rendered train frame'+view.image_name,flush=True)
    assert_frozen(repo)
    summary['max_cached_pose_error']=max_pose_error
    summary['frozen_C1_RGB_pixel_match_all12']=True


def plots_and_verify(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw,ImageFont
    for info in summary['candidates']:
        folder=OUTPUT/info['name'];centers=np.load(folder/'center_trajectories.npz')['centers'];p=palette(info['actual_groups'])
        fig,axes=plt.subplots(1,3,figsize=(13,3.5),constrained_layout=True)
        for group in sorted(info['groups'],key=lambda g:g['kernels'],reverse=True)[:24]:
            g=group['id'];c=centers[:,g]
            axes[0].plot(c[:,0],c[:,1],'-o',ms=2,color=p[g],label=f'G{g}')
            axes[1].plot(c[:,0],c[:,2],'-o',ms=2,color=p[g])
            axes[2].plot(summary['training_frame_ids'],np.linalg.norm(c-c[0],axis=-1),'-o',ms=2,color=p[g])
        axes[0].set(xlabel='world x',ylabel='world y',title='Center trajectory XY')
        axes[1].set(xlabel='world x',ylabel='world z',title='Center trajectory XZ')
        axes[2].set(xlabel='Training frame',ylabel='Center displacement',title='Existing C1 predictions (not trained centers)')
        for ax in axes:ax.grid(alpha=.2)
        if info['actual_groups']:axes[0].legend(ncol=3,fontsize=7)
        fig.savefig(folder/'center_trends.png',dpi=140);plt.close(fig)
    checked=0
    for index in range(12):
        with Image.open(PREVIOUS/'canonical_K5'/f'{index:05d}.png') as ref:
            expected=np.asarray(ref)[50:500,:1600]
        for info in summary['candidates']:
            with Image.open(OUTPUT/info['name']/f'{index:05d}.png') as im:
                assert im.size==(2400,545) and np.array_equal(np.asarray(im)[50:500,:1600],expected)
            checked+=1
        for k in (5,6):
            with Image.open(OUTPUT/f'comparison_baseK{k}'/f'{index:05d}.png') as im:
                assert im.size==(4000,500) and np.array_equal(np.asarray(im)[50:500,:1600],expected)
                for column,info in enumerate((c for c in summary['candidates'] if c['base_K']==k),start=2):
                    with Image.open(OUTPUT/info['name']/f'{index:05d}.png') as triptych:
                        assert np.array_equal(np.asarray(im)[50:500,column*800:(column+1)*800],
                                              np.asarray(triptych)[50:500,1600:2400])
            checked+=1
    for path,expected in summary['source_sha256'].items():
        if digest(ROOT/path)!=expected:raise ValueError('C1 source changed')
    if digest(PREVIOUS/'kernel_evidence.npz')!=summary['evidence_sha256']:
        raise ValueError('Cached evidence changed during review')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    for index in (0,8,11):
        canvas=Image.new('RGB',(1920,780),'#edf2f7');d=ImageDraw.Draw(canvas)
        for j,info in enumerate(summary['candidates']):
            x,y=(j%3)*640,(j//3)*390
            with Image.open(OUTPUT/info['name']/f'{index:05d}.png') as im:
                panel=im.crop((1600,50,2400,500)).resize((640,360),Image.Resampling.LANCZOS)
            d.text((x+6,y+4),info['name']+f" ({info['actual_groups']})",font=font,fill='#17304a')
            canvas.paste(panel,(x,y+30))
        canvas.save(OUTPUT/f"overview_frame{summary['training_frame_ids'][index]:04d}.png")
    code=[Path(__file__),ROOT/'variants/object_centers/motion_grouping.py',ROOT/'tests/test_motion_grouping.py']
    write(OUTPUT/'qa.json',{'checked_PNGs':checked,'GT_C1_pixel_match_previous_review':True,
        'all_three_group_panels_match_five_column_comparisons':True,
        'source_hashes_match':True,'max_cached_pose_error':summary['max_cached_pose_error'],
        'code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in code},
        'training_integration':'none','test_RGB_used':False})


def gallery(summary):
    data=json.dumps(summary,ensure_ascii=False)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C1: three motion grouping methods</title><style>body{margin:24px;background:#142033;color:#eef5ff;font:16px system-ui}a{color:#9be0b8}select,button{padding:9px;margin:6px}input{width:30%}img{width:100%;height:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #53677d}.note{padding:16px;background:#403a2a}</style>
<h1>C1：三种运动聚类对比</h1><p>GT | C1 render | 分组着色。只使用12个训练时刻；未训练新模型。分组身份固定，但不是物体真值。</p>
<p>T1：纯位移轨迹；T2：位移＋轨迹差分速度；T3：在T2运动组内按跨时刻保持邻近的空间图细分。前两种特征不包含RGB或绝对位置。每种比较baseK=5/6；T3最终组数由连通分量决定，小分量变为未知。</p>
<p class="note" id="finding">候选需要人工检查，不能把聚类颜色直接当作真实物体实例。灰色表示未分配，亮色替换可能显露原本暗色的kernel。</p>
<p><a href="../c1_final50000/index.html">上一轮六种候选</a> | <a href="overview_frame0002.png">frame2总览</a> | <a href="overview_frame0146.png">frame146总览</a> | <a href="overview_frame0200.png">frame200总览</a> | <a href="qa.json">校验凭据</a></p>
<select id="candidate"></select><button id="prev">上一帧</button><input id="frame" type="range" min="0" max="11" value="0"><button id="next">下一帧</button><span id="position"></span> <a id="png">打开原图</a>
<img id="image"><p id="counts"></p><h2>三种方法与GT/C1同图比较</h2><p id="comparisonLink"></p><img id="comparison">
<h2>中心轨迹（已有C1轨迹统计，最多显示24组）</h2><img id="trends"><p><a id="ids">固定kernel ID</a> | <a href="summary.json">完整结果</a></p>
<table><thead><tr><th>组</th><th>kernel数</th><th>中心位移RMS</th><th>组内相对位移RMS</th></tr></thead><tbody id="rows"></tbody></table>
<script>const s=DATA,c=document.getElementById('candidate'),f=document.getElementById('frame');s.candidates.forEach(i=>{let o=document.createElement('option');o.value=i.name;o.textContent=i.name+' / '+i.actual_groups+' groups';c.appendChild(o)});function update(){const i=s.candidates.find(i=>i.name===c.value),n=String(f.value).padStart(5,'0'),p=i.name+'/'+n+'.png',q='comparison_baseK'+i.base_K+'/'+n+'.png';document.getElementById('image').src=p;document.getElementById('png').href=p;document.getElementById('comparison').src=q;document.getElementById('comparisonLink').innerHTML='<a href="'+q+'">打开五列原图</a>';document.getElementById('trends').src=i.name+'/center_trends.png';document.getElementById('ids').href=i.name+'/kernel_group_ids.npy';document.getElementById('position').textContent='train frame '+s.training_frame_ids[+f.value]+' / '+(+f.value+1)+'/12';document.getElementById('counts').textContent=i.actual_groups+' groups | '+i.assigned_kernels+'/'+i.total_kernels+' assigned | '+i.unassigned_kernels+' unknown';document.getElementById('rows').innerHTML=i.groups.map(g=>'<tr><td style="color:rgb('+g.color_rgb.map(v=>Math.round(v*255)).join(',')+')">G'+g.id+'</td><td>'+g.kernels+'</td><td>'+g.center_motion_rms.toFixed(4)+'</td><td>'+g.within_group_displacement_rms.toFixed(4)+'</td></tr>').join('');if(s.visual_review)document.getElementById('finding').textContent=s.visual_review;}function step(n){f.value=Math.max(0,Math.min(11,+f.value+n));update()}c.onchange=f.oninput=update;document.getElementById('prev').onclick=()=>step(-1);document.getElementById('next').onclick=()=>step(1);update();</script></html>'''
    (OUTPUT/'index.html').write_text(html.replace('DATA',data),encoding='utf-8')
    old_page=PREVIOUS/'index.html'
    if old_page.exists():
        old=old_page.read_text(encoding='utf-8')
        if 'id="motion-methods-link"' not in old:
            link='<p id="motion-methods-link"><a href="../motion_methods_final50000/index.html">新增对比：纯轨迹 / 轨迹＋速度 / 运动＋空间连通</a></p>'
            old_page.write_text(old.replace('</h1>','</h1>'+link,1),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    summary=prepare()
    if args.prepare_only:return
    render_candidates(summary);plots_and_verify(summary)
    summary['status']='completed';write(OUTPUT/'summary.json',summary)
    write(REPORTS/'motion_grouping_review.json',summary);gallery(summary)
    print('Completed gallery: '+str(OUTPUT/'index.html'),flush=True)


if __name__=='__main__':main()
