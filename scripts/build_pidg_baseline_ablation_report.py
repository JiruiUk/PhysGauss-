"""Create matching Chinese and English reports on the fixed 50 official test views.

Saved PNG metrics only; no training and no modification of experiment data.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
import pidg_ablation_report_layout as layout

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'
TMP=ROOT/'tmp/pdfs/pidg_official_reports'
METRICS=('PSNR','SSIM','LPIPS','MS-SSIM')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_data():
    raw_mlp=read(ROOT/'experiments/mlp_capacity/results.json')
    raw_mask=read(ROOT/'experiments/random_frame_mask/results.json')
    assert raw_mlp['controller']['status']==raw_mask['controller']['status']=='completed'
    mlp=[{k:r[k] for k in ('id','width','depth','mlp_parameters','metrics','run')} for r in raw_mlp['rows']]
    mask=[{k:r[k] for k in ('drop_percent','retained','removed','official_metrics','run')} for r in raw_mask['rows']]
    assert [r['id'] for r in mlp]==['W256_D0','W512_D0','W256_D2','W512_D2']
    assert [r['drop_percent'] for r in mask]==[0,10,30,50,70,90]
    test_path=ROOT/'data/PIDG/rubber_duck/transforms_test.json'
    test=read(test_path)
    assert len(test['frames'])==50
    catalog=[]
    for family,rows in [('mlp',mlp),('mask',mask[1:])]:
        for row in rows:
            run=Path(row['run']); info=read(run/'reproduction.json')
            assert info['status']=='completed'
            assert (info['iterations'],info['resolution_width'],info['seed'])==(50000,1600,0)
            metric=row['metrics'] if family=='mlp' else row['official_metrics']
            assert metric==read(run/'results.json')['ours_best']
            assert all(math.isfinite(metric[k]) for k in METRICS)
            raw=run/'test/ours_best'; expected={f'{i:05d}.png' for i in range(50)}
            for kind in ('renders','gt'):
                assert {p.name for p in (raw/kind).glob('*.png')}==expected
            per=read(run/'per_view.json')['ours_best']
            for key in METRICS:
                assert set(per[key])==expected
                assert abs(math.fsum(per[key].values())/50-metric[key])<1e-5
            recorded_hash=info.get('dataset_hashes',{}).get('transforms_test.json')
            if recorded_hash and family=='mlp':
                assert recorded_hash==digest(test_path)
            if family=='mask':
                overlay=ROOT/f"data/pidg_random_frame_mask/rubber_duck/drop{row['drop_percent']:02d}_seed0/dataset/transforms_test.json"
                assert read(overlay)==test
                assert recorded_hash==digest(overlay)
            row['best_iteration']=int(re.findall(r'\d+',(run/'deform/iteration_best/iter.txt').read_text())[-1])
            catalog.append({'family':family,'label':row.get('id',f"Drop {row.get('drop_percent')}%"),
                            'split':'official_test','frame_count':50,'best_iteration':row['best_iteration'],
                            'render_absolute':str(raw/'renders'),'gt_absolute':str(raw/'gt'),
                            'reproduction_sha256':digest(run/'reproduction.json'),
                            'results_sha256':digest(run/'results.json'),'per_view_sha256':digest(run/'per_view.json')})
    assert mask[0]['official_metrics']==mlp[0]['metrics']
    mask[0]['best_iteration']=mlp[0]['best_iteration']
    return {'mlp':mlp,'mask':mask,'test':test,'catalog':catalog,'manifest_sha256':digest(test_path)}


def figures(data):
    result={}; checks=[]
    def save(fig,name):
        path=TMP/(name+'.png'); fig.savefig(path,dpi=210,bbox_inches='tight'); plt.close(fig)
        result[name]=path
    mlp,mask=data['mlp'],data['mask']
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.6))
    x=[r['mlp_parameters'] for r in mlp]; y=[r['metrics']['PSNR'] for r in mlp]
    axes[0].plot(x,y,'o-',color='#087F8C'); axes[0].set_xscale('log'); axes[0].set_ylim(27,29)
    axes[0].set_xlabel('Deformation MLP parameters (log scale)'); axes[0].set_ylabel('Official-test PSNR (dB)')
    for i,row in enumerate(mlp):
        axes[0].annotate(row['id'],(x[i],y[i]),xytext=(-80,8) if i==3 else (6,5),textcoords='offset points',fontsize=8.5)
    axes[1].bar(['256 / 0','512 / 0','256 / 2','512 / 2'],[r['metrics']['LPIPS'] for r in mlp],color=['#8395A5','#087F8C','#2D6D91','#BA8756'])
    axes[1].set_ylim(0,.04); axes[1].set_ylabel('Official-test LPIPS (lower is better)'); axes[1].set_xlabel('Width / depth')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    fig.tight_layout(w_pad=3); save(fig,'mlp_chart')
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.6)); x=[r['drop_percent'] for r in mask]
    for ax,key,label in zip(axes,['PSNR','LPIPS'],['Official-test PSNR (dB)','Official-test LPIPS']):
        y=[r['official_metrics'][key] for r in mask]
        ax.plot(x,y,'o-',color='#087F8C'); ax.axhline(y[0],color='#8395A5',ls='--',lw=.8)
        ax.set_xticks(x); ax.set_xlabel('Training frames removed (%)'); ax.set_ylabel(label)
        ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.15)
    fig.tight_layout(w_pad=3); save(fig,'mask_chart')
    def comparison(rows,labels,name,columns):
        gt=Image.open(Path(rows[0]['run'])/'test/ours_best/gt/00025.png').convert('RGB')
        assert gt.size==(1600,900)
        yy,xx=np.where(np.array(gt).max(axis=2)>12)
        roi=(max(0,int(xx.min())-24),max(0,int(yy.min())-24),min(1600,int(xx.max())+25),min(900,int(yy.max())+25))
        pw,gap,lh=960,24,82; ph=round(pw*(roi[3]-roi[1])/(roi[2]-roi[0]))
        entries=[{'label':'GT','path':Path(rows[0]['run'])/'test/ours_best/gt/00025.png','subtitle':'same camera and time'}]
        panel_checks=[]
        for row,label in zip(rows,labels):
            raw=Path(row['run'])/'test/ours_best'; gt_path=raw/'gt/00025.png'; pred=raw/'renders/00025.png'
            assert Image.open(gt_path).convert('RGB').tobytes()==gt.tobytes()
            entries.append({'label':label,'path':pred,'subtitle':f"best @ {row['best_iteration']:,}"})
            panel_checks.append({'label':label,'render':str(pred),'gt':str(gt_path),'render_sha256':digest(pred),'gt_sha256':digest(gt_path)})
        nrows=(len(entries)+columns-1)//columns
        image=Image.new('RGB',(columns*pw+(columns-1)*gap,nrows*(ph+lh)+(nrows-1)*gap),'#eff5f8'); draw=ImageDraw.Draw(image)
        for i,e in enumerate(entries):
            x=(i%columns)*(pw+gap); y=(i//columns)*(ph+lh+gap)
            draw.text((x+12,y+14),e['label'],font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',45),fill='#153149')
            draw.text((x+pw-360,y+24),e['subtitle'],font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',33),fill='#637487')
            source=Image.open(e['path']).convert('RGB'); assert source.size==gt.size
            image.paste(source.crop(roi).resize((pw,ph),Image.Resampling.LANCZOS),(x,y+lh))
        path=TMP/(name+'.png'); image.save(path); result[name]=path
        checks.append({'figure':name,'official_test_index':25,'roi_xyxy':roi,'gt_pixel_identity':True,'panels':panel_checks})
    comparison(mlp,[r['id'] for r in mlp],'mlp_visual',3)
    low=[r for r in mask if r['drop_percent'] in (0,10,30)]
    high=[r for r in mask if r['drop_percent'] in (50,70,90)]
    comparison(low,['Baseline 0%','Drop 10%','Drop 30%'],'low_visual',2)
    comparison(high,['Drop 50%','Drop 70%','Drop 90%'],'high_visual',2)
    result['checks']=checks
    return result


def build(lang,data,fig,time):
    q=lambda cn,en:cn if lang=='cn' else en
    for name,file in [('CN','msyh.ttc' if lang=='cn' else 'arial.ttf'),('CN-Bold','msyhbd.ttc' if lang=='cn' else 'arialbd.ttf'),('Arial','arial.ttf'),('PathFont','consola.ttf')]:
        pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file,subfontIndex=0))
    pdfmetrics.registerFontFamily('CN',normal='CN',bold='CN-Bold',italic='CN',boldItalic='CN-Bold')
    for name,style in layout.styles.items():
        style.wordWrap='CJK' if lang=='cn' or name=='path' else 'LTR'
        if name=='body': style.leading=17 if lang=='cn' else 15.5
    pdf=OUT/('PIDG_MLP_MaskedData_Report'+('' if lang=='cn' else '_EN')+'.pdf')
    layout.c=canvas.Canvas(str(pdf),pagesize=(layout.W,layout.H),pageCompression=1)
    layout.c.setTitle(q('PIDG：MLP容量与缺帧实验 - 固定官方测试集','PIDG: MLP Capacity and Missing Frames - Fixed Official Test Set'))
    layout.c.setAuthor('PhysGauss / PIDG Extension Project')
    layout.c.setSubject('Fixed 50-view official-test comparisons; frozen best snapshots; no training')
    layout.REPORT_TIME=time; layout.page_records=[]
    def page(n,title_cn,title_en,sub_cn='',sub_en=''):
        return layout.Page(n,q('固定官方测试 / 50张','Fixed official test / 50 views'),q(title_cn,title_en),q(sub_cn,sub_en))
    def visual(p,key,caption_cn,caption_en,height=430):
        im=Image.open(fig[key]); p.image(fig[key],min(height,layout.CW*im.height/im.width),q(caption_cn,caption_en))
    mlp,mask=data['mlp'],data['mask']; base=mlp[0]['metrics']; cw=layout.CW

    p=page(1,'PIDG：MLP容量与随机缺帧实验','PIDG: MLP Capacity and Missing Frames',
           '固定50张官方测试图 | '+time.strftime('%Y-%m-%d %H:%M %Z'),'Fixed 50-view official test | '+time.strftime('%Y-%m-%d %H:%M %Z'))
    p.callout(q('<b>统一评价范围：</b>全部 PSNR、SSIM、LPIPS、MS-SSIM 均来自相同的50张官方测试图。每张图先计算指标，再取算术平均。',
                '<b>One evaluation set:</b> every PSNR, SSIM, LPIPS and MS-SSIM result uses the same 50 official test images. Metrics are computed per image and averaged.'))
    p.text(q('两组均使用原始 PIDG 兼容 baseline：Rubber Duck、50k训练步、1600×900渲染、训练seed0。容量实验改变形变MLP；缺帧实验使用原始W256_D0并随机移除训练图像。',
             'Both studies use the original PIDG compatibility baseline: Rubber Duck, 50k iterations, 1600×900 renders and training seed0. The capacity study changes the deformation MLP; the removal study keeps original W256_D0 and removes training images.'))
    p.callout(q('<b>容量：</b>W512_D0 的 PSNR 最高，为28.577 dB，比baseline高1.320 dB。更大容量未带来单调提升；原始baseline的LPIPS最低。',
                '<b>Capacity:</b> W512_D0 has the highest PSNR, 28.577 dB, a 1.320 dB gain over baseline. Larger capacity is not monotonically better; original baseline has the lowest LPIPS.'))
    p.callout(q('<b>训练缺帧：</b>10%移除为28.253 dB；90%移除降至23.063 dB，比完整训练数据baseline低4.194 dB。中间各档并非严格单调。',
                '<b>Training-frame removal:</b> 10% yields 28.253 dB; 90% falls to 23.063 dB, 4.194 dB below the complete-training-data baseline. Intermediate ratios are not strictly monotonic.'))
    p.text(q('单场景、单训练种子、单缺帧种子；复用历史baseline。以上为描述性结果，尚未证明统计显著性。',
             'One scene, one training seed and one removal seed; historical baseline reused. These are descriptive results without a statistical significance claim.'),'small')
    p.text(q('索引：结构02；协议03；MLP结果04-05；缺帧结果06-08；快照与来源09；render / GT绝对地址10-11。',
             'Guide: architecture 02; protocol 03; MLP results 04-05; removal results 06-08; snapshots and sources 09; absolute render / GT paths 10-11.'),'small'); p.finish()

    p=page(2,'形变MLP究竟修改了什么','What Changes in the Deformation MLP?',
           'width控制分支宽度；depth控制融合后的额外共享隐藏层数。','Width controls branch size; depth counts extra shared hidden layers after fusion.')
    p.table([[q('模块','Module'),q('结构','Structure')],
             [q('空间分支','Spatial branch'),'XYZ hash: 32 → Linear(32, w) → 2×Sigmoid−1'],
             [q('时空分支','Temporal branch'),'XYT / YZT / XZT hash: 192 → Linear(192, w) → ReLU'],
             [q('融合','Fusion'),q('两分支逐元素相乘，得到w维特征','Elementwise product of both branches; w features')],
             [q('额外共享层','Extra shared layers'),q('D0：无；D2：两层 Linear(w, w) → ReLU','D0: none; D2: two Linear(w, w) → ReLU blocks')],
             [q('三个输出头','Three output heads'),q('w→7：位置变换（四元数4+平移3）；w→4：旋转偏移；w→3：尺度偏移',
                                                  'w→7: positional warp (quaternion (4) + translation (3)); w→4: rotation offset; w→3: scale offset')]], [110,cw-110])
    p.table([[q('配置','Model'),'w / d',q('MLP参数量','MLP parameters'),q('相对baseline','Relative size')],
             *[[r['id'],f"{r['width']} / {r['depth']}",f"{r['mlp_parameters']:,}",f"{r['mlp_parameters']/61454:.2f}×"] for r in mlp]], [112,83,155,cw-350])
    p.text(q('<b>D0仍有网络：</b>保留两条输入分支和三个线性输出头，仅在融合后不添加额外隐藏层。W512_D0把256维加宽至512维。参数量不包含哈希网格。',
             '<b>D0 still has a network:</b> both input branches and all three linear output heads remain, with no extra hidden layer after fusion. W512_D0 widens 256 features to 512. Counts exclude hash grids.'))
    p.text(q('哈希网格、Gaussian、渲染器、速度/应力网络保持一致。原速度/应力MLP为共享43→64→64主干，输出速度3和应力6。新增F/NH/材料学习/结构化外力分支关闭。',
             'Hash grids, Gaussians, renderer and velocity / stress network are unchanged. The original velocity / stress MLP has a shared 43→64→64 trunk with 3D velocity and 6-component stress heads. New F / NH / material / structured-force branches are disabled.'),'small'); p.finish()

    p=page(3,'固定测试集，改变训练数据量','Fixed Test Set, Different Training Subsets',
           '缺帧是随机移除整张训练图像；mask seed0，嵌套子集。','Removal drops whole training images, not pixels; removal seed0, nested subsets.')
    p.table([[q('移除比例','Removed'),q('保留训练帧','Training kept'),q('移除训练帧','Training removed'),q('评价图像数','Test images')],
             *[[f"{r['drop_percent']}%",r['retained'],r['removed'],50] for r in mask]], [87,139,146,cw-372])
    p.text(q('原始训练集150帧，官方测试集固定50帧。各比例只改变训练清单；图像、时间、位姿和规范点云保留。高比例包含低比例已移除图像，保留原顺序。',
             'The original split has 150 training frames and 50 fixed official test frames. Only the training manifest changes. Images, times, poses and canonical points remain. Higher removal includes earlier removed frames; original order is preserved.'))
    p.callout(q('<b>测试集使用：</b>保留训练帧参与loss与反向传播。官方测试图不参与梯度更新，但训练期间会定期评估，并按测试PSNR选择ours_best。因此测试集也承担模型选择功能。',
                '<b>Test-set usage:</b> retained training images supply the loss and gradients. Official test images do not supply gradient updates, but are evaluated during training to select ours_best by test PSNR. The test set therefore also serves model selection.'))
    p.text(q('只比较固定官方测试集上best快照的PNG渲染指标。新运行均训练至50k，best可能来自更早迭代。PSNR、SSIM、MS-SSIM越高越好；LPIPS越低越好。',
             'Only best-snapshot PNG metrics on the fixed official test set are compared. New runs train to 50k; best may come from an earlier iteration. Higher PSNR, SSIM and MS-SSIM are better; lower LPIPS is better.'))
    p.text(q('固定50k预算使高缺帧时每张保留图重复更多次。原PIDG动量查询fid/train_count的分母也随训练帧数改变，本轮沿用原生行为。PSNR取逐图dB均值。',
             'A fixed 50k budget increases image reuse at high removal. The denominator of native PIDG momentum queries, fid/train_count, also changes with training-set size; original behavior is preserved. PSNR averages per-image dB values.'),'small'); p.finish()

    p=page(4,'加宽有收益，容量收益不单调','Wider Helps; Capacity Gains Are Not Monotonic',
           '相同50张官方测试图；best快照PNG渲染。','The same 50 official test images; best-snapshot PNG renders.')
    p.table([[q('配置','Model'),q('MLP参数','Parameters'),'PSNR','ΔPSNR','SSIM','LPIPS','MS-SSIM'],
             *[[r['id'],f"{r['mlp_parameters']:,}",f"{r['metrics']['PSNR']:.3f}",f"{r['metrics']['PSNR']-base['PSNR']:+.3f}",
                *[f"{r['metrics'][k]:.6f}" for k in ('SSIM','LPIPS','MS-SSIM')]] for r in mlp]], [82,72,57,59,77,78,cw-425],highlight=2)
    visual(p,'mlp_chart','图1  容量与PSNR / LPIPS。连线仅辅助阅读，参数横轴为对数刻度。',
           'Figure 1. Capacity versus PSNR / LPIPS. Connecting lines guide reading; the parameter axis is logarithmic.',205)
    p.text(q('W512_D0比baseline高1.320 dB。在宽度512下再加两层共享隐藏层，PSNR下降0.961 dB。最大模型未取得最高PSNR。',
             'W512_D0 gains 1.320 dB over baseline. At width 512, adding two shared hidden layers reduces PSNR by 0.961 dB. The largest model does not achieve the highest PSNR.'))
    p.text(q('W256_D0的LPIPS最低，W512_D2的MS-SSIM最高，排序依赖评价指标。历史baseline缺少完整源码哈希；单种子结果不能证明稳定因果收益。',
             'W256_D0 has the lowest LPIPS; W512_D2 has the highest MS-SSIM. Rankings depend on the metric. Historical baseline lacks complete source hashes; one seed cannot establish stable causal gains.'),'small'); p.finish()

    frame=data['test']['frames'][25]
    view=q('官方测试索引25 | 源图像 ','Official test index 25 | source ')+frame['file_path']+f" | t={frame['time']:.3f}"
    p=page(5,'MLP容量：同一视图GT对照','MLP Capacity: GT and Four Models',view,view)
    p.text(q('同一GT裁剪窗口与等比例缩放，仅用于展示。没有颜色增强；指标使用完整1600×900 PNG。',
             'The same GT-derived crop and proportional resizing are used for display only. No color enhancement; metrics use complete 1600×900 PNGs.'),'small')
    visual(p,'mlp_visual','图2  GT与四种形变MLP，标签标记best步数；末尾空白仅为排版。',
           'Figure 2. GT and four deformation MLPs, labelled by best iteration. The final empty cell is a layout choice.',425)
    p.text(q('对照鸭体轮廓、遮挡与支架边缘。单图不能代替50图均值。MLP render与GT绝对地址见第10页。',
             'Compare contours, occlusions and support edges. One view cannot replace the 50-image mean. Absolute MLP render and GT directories are on page 10.')); p.finish()

    p=page(6,'训练缺帧：90%时明显下降','Severe Removal Reduces Reconstruction Quality',
           '全部使用W256_D0；评价始终为相同50张官方测试图。','Every model uses W256_D0; evaluation always uses the same 50 official test images.')
    p.table([[q('缺帧','Removed'),q('保留训练帧','Train kept'),'PSNR','ΔPSNR','SSIM','LPIPS','MS-SSIM'],
             *[[f"{r['drop_percent']}%",r['retained'],f"{r['official_metrics']['PSNR']:.3f}",f"{r['official_metrics']['PSNR']-base['PSNR']:+.3f}",
                *[f"{r['official_metrics'][k]:.6f}" for k in ('SSIM','LPIPS','MS-SSIM')]] for r in mask]], [58,78,58,59,77,78,cw-408],highlight=2)
    visual(p,'mask_chart','图3  同一官方测试集上的缺帧曲线；虚线为完整训练数据baseline。',
           'Figure 3. Removal curves on the same official test set; dashed lines mark the complete-training-data baseline.',200)
    p.text(q('10%缺帧比历史baseline高0.996 dB；中间比例不严格单调。90%缺帧比baseline低4.194 dB，LPIPS从0.029624升至0.060427。',
             '10% removal exceeds historical baseline by 0.996 dB; intermediate ratios are not strictly monotonic. At 90%, PSNR is 4.194 dB below baseline and LPIPS rises from 0.029624 to 0.060427.'))
    p.text(q('10%的单次增益仍需同环境、多训练种子与多缺帧种子验证；不能据此断言删除训练图像通常有益。六个比例的评价图像集合完全相同。',
             'The single-run 10% gain needs matched-environment repeats over training and removal seeds. It does not establish that deleting training images is generally beneficial. Test membership is identical across all six ratios.'),'small'); p.finish()

    for number,key,title_cn,title_en in [(7,'low_visual','官方测试图像：0%、10%、30%','Official-Test Images: 0%, 10%, 30%'),
                                          (8,'high_visual','官方测试图像：50%、70%、90%','Official-Test Images: 50%, 70%, 90%')]:
        p=page(number,title_cn,title_en,view,view)
        p.text(q('GT与各模型使用相同相机、时间和裁剪窗口；裁剪仅用于展示，指标使用完整图像。',
                 'GT and models use the same camera, time and crop. Cropping is display-only; metrics use complete images.'),'small')
        visual(p,key,f'图{number-3}  固定官方测试视图；面板标记各模型best步数。',
               f'Figure {number-3}. A fixed official test view, labelled by each model\'s best iteration.',450)
        p.text(q('对照鸭体形状、遮挡、支架边缘和细节。数值结论以第6页50图均值为准。缺帧模型绝对地址见第11页；0%对照见第10页。',
                 'Compare shape, occlusion, support edges and detail. Numerical conclusions use the 50-image means on page 6. Absolute removal-model paths are on page 11; the 0% control is on page 10.'),'small'); p.finish()

    p=page(9,'快照、解释边界与来源','Snapshots, Limits and Sources',
           '50k为训练完成步数；下表为实际渲染的best快照。','50k is the completed training budget; the table lists the best snapshots actually rendered.')
    snapshots=[[r['id'],f"{r['best_iteration']:,}"] for r in mlp]
    snapshots += [[q(f"缺帧{r['drop_percent']}%",f"Removal {r['drop_percent']}%"),f"{r['best_iteration']:,}"] for r in mask[1:]]
    p.table([[q('配置','Model'),q('best迭代步数','Best iteration')],*snapshots],[280,cw-280])
    p.text(q('<b>边界：</b>90%缺帧仅保留15张训练图，时间覆盖约0.067至0.919；官方测试还含区间外时刻，故也包含时间外推。规范点云仍共享。',
             '<b>Limits:</b> 90% removal leaves 15 training images covering roughly t=0.067 to 0.919. Official testing includes times outside that interval, so temporal extrapolation contributes. Canonical points remain shared.'),'small')
    p.text(q('<b>可比性：</b>所有结果均使用相同50张官方测试图。复用历史baseline；训练seed0、缺帧seed0。best由官方测试PSNR选择，报告测试集未独立于模型选择；未提供统计显著性。',
             '<b>Comparability:</b> all results use the same 50 official test images. Historical baseline reused; training seed0 and removal seed0. Official-test PSNR selects best, so the reported test set is not independent of model selection. Statistical significance is not established.'),'small')
    p.heading(q('可复核的本地来源','Local sources for verification'))
    for line in ['experiments/mlp_capacity/{results.json,plan.json,validation.json}',
                 'experiments/random_frame_mask/{results.json,plan.json,preflight.json}',
                 'third_party/PIDG/scene_PIDG/network.py; train_pidg.py; render.py; metrics.py',
                 'data/PIDG/rubber_duck/transforms_test.json; run files: reproduction.json, results.json, per_view.json']:
        p.text(escape(line),'small',gap=5)
    p.text(q('配套sources.json记录官方测试指标、best步数、图片路径与来源哈希。9个唯一render目录各含50张PNG与匹配GT。',
             'Companion sources.json records official-test metrics, best iterations, image paths and source hashes. All 9 unique render directories contain 50 PNGs and matching GT.'),'small'); p.finish()

    for number,family,title_cn,title_en in [(10,'mlp','MLP实验：render与GT绝对地址','MLP Study: Absolute Render and GT Paths'),
                                            (11,'mask','缺帧实验：render与GT绝对地址','Removal Study: Absolute Render and GT Paths')]:
        p=page(number,title_cn,title_en,'均为固定50张官方测试图；路径换行仅为排版，不省略字符。',
               'All directories contain the fixed 50 official test views. Line breaks are layout only; no path characters are omitted.')
        for e in [x for x in data['catalog'] if x['family']==family]:
            p.heading(e['label']+' | best '+f"{e['best_iteration']:,}")
            p.text('Render:','small',gap=2); p.text(layout.path_markup(e['render_absolute']),'path',gap=6)
            p.text('GT:','small',gap=2); p.text(layout.path_markup(e['gt_absolute']),'path',gap=10)
        p.text(q('每个目录为00000.png至00049.png；render与GT文件名对应，分辨率1600×900。',
                 'Each directory contains 00000.png through 00049.png. Render and GT filenames match at 1600×900 resolution.'),'small')
        if family=='mask':
            p.text(q('0%缺帧复用W256_D0，官方测试地址见第10页第一项。',
                     '0% removal reuses W256_D0; its official-test paths are the first entry on page 10.'),'small')
        p.finish()
    assert len(layout.page_records)==11
    layout.c.save()
    sources=['experiments/mlp_capacity/results.json','experiments/mlp_capacity/plan.json',
             'experiments/random_frame_mask/results.json','experiments/random_frame_mask/plan.json',
             'data/PIDG/rubber_duck/transforms_test.json','third_party/PIDG/scene_PIDG/network.py',
             'third_party/PIDG/train_pidg.py','third_party/PIDG/render.py','third_party/PIDG/metrics.py',
             'scripts/build_pidg_baseline_ablation_report.py','scripts/pidg_ablation_report_layout.py']
    archive={'pdf':str(pdf),'language':lang,'created_london':time.isoformat(),
             'evaluation_scope':'fixed_50_official_test_views_only','test_frame_count':50,
             'aggregation':'arithmetic mean of per-image metrics; PSNR in dB',
             'snapshot_selection':'historical best official-test PSNR; test images do not supply training gradients',
             'pages':layout.page_records,'mlp_rows':mlp,'mask_rows':mask,'render_catalog':data['catalog'],
             'unique_render_directories':9,'total_render_frames':450,'image_checks':fig['checks'],
             'official_manifest_sha256':data['manifest_sha256'],'source_sha256':{s:digest(ROOT/s) for s in sources}}
    pdf.with_suffix('.sources.json').write_text(json.dumps(archive,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'pdf':str(pdf),'language':lang,'pages':11,'scope':archive['evaluation_scope'],
                      'content_bottom':[p['content_bottom'] for p in layout.page_records]}))


def main():
    OUT.mkdir(parents=True,exist_ok=True); TMP.mkdir(parents=True,exist_ok=True)
    data=load_data(); fig=figures(data)
    time=datetime.now(timezone.utc).astimezone(ZoneInfo('Europe/London'))
    for language in ('cn','en'):
        build(language,data,fig,time)


if __name__=='__main__':
    main()
