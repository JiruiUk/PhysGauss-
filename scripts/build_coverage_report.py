"""Chinese C0-C5 report from verified final PNG metrics and training logs."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import re
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
import pidg_ablation_report_layout as L

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'
ASSETS=OUT/'coverage_assets'
PDF=OUT/'PIDG_Coverage_Ablation_Report_CN.pdf'
MD=PDF.with_suffix('.md')
ASSESS=ROOT/'experiments/coverage_isolated/assessment.json'
METRICS=['PSNR','SSIM','LPIPS','MS-SSIM']
NAMES=['C0_V3_reference','C1_mask','C2_mask_affine','C3_mask_affine_density','C4_affine_only','C5_density_only']
ROOTS=[ROOT/'runs/pidg_coverage_tuning']*4+[ROOT/'runs/pidg_coverage_isolated']*2
COLORS=['#596a7c','#d9862e','#198f86','#b55555','#8c6cbb','#3287bf']
SOURCES={}; RECORD=[]


def read(path):
    path=Path(path); SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load():
    assert read(ROOT/'runs/pidg_coverage_tuning/controller.json')['status']=='completed'
    assert read(ROOT/'runs/pidg_coverage_isolated/controller.json')['status']=='completed'
    metadata=read(ROOT/'data/derived/kinematic_resplit/final_test/transforms_test.json')
    assert [int(Path(f['file_path']).stem) for f in metadata['frames']]==list(range(1,201,4))
    models=[]; expected={f'{i:05d}.png' for i in range(50)}
    for name,root in zip(NAMES,ROOTS):
        d=root/name/'rubber_duck'; scores=read(d/'results.json')['ours_50000']
        pv=read(d/'per_view.json')['ours_50000']; coverage=read(d/'coverage_metrics_test.json')['50000']
        assert coverage['count']==50
        for k in METRICS:
            assert set(pv[k])==expected and all(math.isfinite(v) for v in pv[k].values())
            assert abs(math.fsum(pv[k].values())/50-scores[k])<1e-5
        if name.startswith('C0'):
            reference=read(d/'reference.json'); init=reference['initialization']['sha256']
        else:
            assert read(d/'checkpoints/resume.json')['iteration']==50000
            branch=read(d/'branch.json'); assert branch['iteration']==0 and branch['optimizer_updates']==0
            init=branch['identical_full_checkpoint_sha256']
        config=read(d/'metric_config.json'); best=read(d/'metric_best.json')
        assert all(v['selection_split']=='validation' for v in best.values())
        history=[json.loads(s) for s in (d/'metric_history.jsonl').read_text().splitlines()]
        SOURCES[str((d/'metric_history.jsonl').relative_to(ROOT))]=hashlib.sha256((d/'metric_history.jsonl').read_bytes()).hexdigest()
        with (d/'point_cloud/iteration_50000/point_cloud.ply').open('rb') as stream:
            for line in stream:
                if line.startswith(b'element vertex'): count=int(line.split()[-1]);break
        events=[json.loads(s) for s in (d/'coverage_density.jsonl').read_text().splitlines()] if (d/'coverage_density.jsonl').exists() else []
        gallery=read(root/'render_comparisons'/name/'manifest.json')
        assert gallery['pixel_identity_verified'] and gallery['pair_count']==250
        models.append({'id':name[:2],'name':name,'directory':d,'scores':scores,'per_view':pv,
            'coverage':coverage,'history':history,'config':config,'best':best,'initial_sha256':init,
            'points':count,'density_events':events,'gallery':gallery})
    assert len({m['initial_sha256'] for m in models})==1
    for i in range(50):
        hashes=[hashlib.sha256((m['directory']/f'test/ours_50000/gt/{i:05d}.png').read_bytes()).hexdigest() for m in models]
        assert len(set(hashes))==1
    return metadata,models


def alpha_diagnostics(metadata,models):
    for m in models:
        rows=[]
        for i,frame in enumerate(metadata['frames']):
            path=m['directory']/f'test/ours_50000/alpha/{i:05d}.png'
            with Image.open(path) as image:
                a=np.asarray(image.convert('L'))/255.; size=image.size
            target=ROOT/'data/derived/kinematic_resplit/final_test'/(frame['file_path']+'.png')
            with Image.open(target) as image:
                gt=np.asarray(image.getchannel('A').resize(size,Image.Resampling.BICUBIC))/255.
            truth,pred=gt>.5,a>.5
            tp=np.count_nonzero(pred&truth); fp=np.count_nonzero(pred&~truth); fn=np.count_nonzero(~pred&truth)
            row={'frame_id':int(Path(frame['file_path']).stem),'precision':tp/max(tp+fp,1),
                'recall':tp/max(tp+fn,1),'background_false_positive_rate':fp/max(np.count_nonzero(~truth),1),
                'predicted_area_fraction':float(pred.mean()),'gt_area_fraction':float(truth.mean()),
                'mask_iou':tp/max(tp+fp+fn,1)}
            rows.append(row)
            SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        m['alpha_diagnostics']={'per_frame':rows,'mean':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='frame_id'}}
        assert abs(m['alpha_diagnostics']['mean']['mask_iou']-m['coverage']['mean']['mask_iou'])<1e-8


def analyze(models):
    comparisons={}
    for a,b in [(1,0),(4,0),(5,0),(2,1),(3,2),(2,0),(3,0)]:
        newer,ref=models[a],models[b]; per={}
        for k in METRICS:
            delta=np.array([newer['per_view'][k][f'{i:05d}.png']-ref['per_view'][k][f'{i:05d}.png'] for i in range(50)])
            per[k]={'mean':float(delta.mean()),'median':float(np.median(delta)),
                'improved_frames':int(np.count_nonzero(delta<0 if k=='LPIPS' else delta>0)),
                'min':float(delta.min()),'max':float(delta.max())}
        comparisons[f'C{a}-C{b}']={'metric_delta':{k:newer['scores'][k]-ref['scores'][k] for k in METRICS},'per_frame':per,
            'coverage_delta':{k:newer['coverage']['mean'][k]-ref['coverage']['mean'][k] for k in newer['coverage']['mean']}}
    prefix={}
    for a,b in [(5,0),(3,2),(4,0),(2,1)]:
        h=[{r['iteration']:r['metrics'] for r in models[j]['history']} for j in (a,b)]
        prefix[f'C{a}-C{b}']={str(i):h[0][i]['PSNR']-h[1][i]['PSNR'] for i in (1000,3000,10000,15000,16000,18000,36000,50000)}
    tail_delta=np.array([models[3]['per_view']['PSNR'][f'{i:05d}.png']-models[2]['per_view']['PSNR'][f'{i:05d}.png'] for i in range(50)])
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol':{'scene':'Rubber Duck','seed':0,
        'split':[120,30,50],'total_steps_each':50000,'snapshot':'final50000','official_resolution':[1600,900],
        'initial_full_state_sha256':models[0]['initial_sha256'],'all_gt_png_bytes_equal':True,
        'four_metric_means_verified':True,'selection':'validation-only','historical_test_exposure':True},
        'models':[{k:v for k,v in m.items() if k not in ('directory','history','per_view')}|{'directory':str(m['directory'])} for m in models],
        'comparisons':comparisons,'pre_activation_validation_psnr_differences':prefix,
        'C3_minus_C2_last8_test_frames':{'frame_ids':list(range(169,201,4)),
            'last8_mean':float(tail_delta[42:].mean()),'first42_mean':float(tail_delta[:42].mean()),
            'last8_contribution_to50_mean':float(tail_delta[42:].sum()/50)},
        'causal_limit':'Trajectories diverge before affine activation and density additions. No shared15k branch or multi-seed replication; final differences are observed run differences, not isolated causal estimates.'}
    return result


def save_plot(fig,name):
    path=ASSETS/(name+'.png'); fig.savefig(path,dpi=190,bbox_inches='tight',facecolor='white'); plt.close(fig)
    return path


def figures(models):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    output={}
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8))
    for m,color in zip(models,COLORS):
        h=m['history']; x=[p['iteration']/1000 for p in h]
        for ax,k in zip(axes,['PSNR','LPIPS']): ax.plot(x,[p['metrics'][k] for p in h],label=m['id'],color=color,lw=1.4)
    for ax,k in zip(axes,['PSNR','LPIPS']):
        ax.set(xlabel='Training steps (k)',ylabel='Validation '+k); ax.grid(alpha=.15); ax.legend(ncol=3,fontsize=8)
    fig.tight_layout();output['curves']=save_plot(fig,'validation_curves')
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.3))
    for ax,(a,b) in zip(axes,[(5,0),(3,2)]):
        h=[{r['iteration']:r['metrics']['PSNR'] for r in models[j]['history']} for j in (a,b)]
        steps=sorted(set(h[0])&set(h[1])); delta=[h[0][s]-h[1][s] for s in steps]
        ax.plot(np.array(steps)/1000,delta,lw=1.7,color=COLORS[a]); ax.axhline(0,c='gray',lw=.7)
        ax.axvspan(0,18,color='#c9d4df',alpha=.35,label='Before first additions')
        ax.axvline(18,c='#bb7431',ls='--',lw=1)
        ax.set(title=f'C{a} - C{b}',xlabel='Training steps (k)',ylabel='Validation PSNR difference (dB)')
        ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.tight_layout();output['prefix']=save_plot(fig,'prefix_differences')
    ids=np.arange(1,201,4)
    fig,ax=plt.subplots(figsize=(10.5,3.6))
    for m,color in zip(models,COLORS):ax.plot(ids,[m['per_view']['PSNR'][f'{i:05d}.png'] for i in range(50)],label=m['id'],color=color,lw=1.3)
    ax.set(xlabel='Original test frame ID',ylabel='Final50k test PSNR (dB)');ax.legend(ncol=6,fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();output['frames']=save_plot(fig,'test_per_frame')
    # This frame was discussed before these runs were compared: original153,
    # export00038. One GT-defined crop shared by every panel, no per-model crop.
    index=38; name=f'{index:05d}.png'
    with Image.open(models[0]['directory']/'test/ours_50000/gt'/name) as im: gt=im.convert('RGB').copy()
    nonzero=np.asarray(gt).max(-1)>5; yy,xx=np.where(nonzero)
    box=(max(0,int(xx.min())-25),max(0,int(yy.min())-25),min(gt.width,int(xx.max())+26),min(gt.height,int(yy.max())+26))
    fig,axes=plt.subplots(2,4,figsize=(10.5,6.5))
    images=[gt]+[Image.open(m['directory']/'test/ours_50000/renders'/name).convert('RGB') for m in models]
    for ax,im,title in zip(axes.flat,images,['GT']+[m['id'] for m in models]):
        ax.imshow(im.crop(box));ax.set_title(title);ax.axis('off')
    error=np.abs(np.asarray(images[2],dtype=float)/255.-np.asarray(gt,dtype=float)/255.).mean(-1)
    ax=axes.flat[-1];ax.imshow(error[box[1]:box[3],box[0]:box[2]],cmap='magma',vmin=0,vmax=.25)
    ax.set_title('C1 RGB abs error [0,0.25]');ax.axis('off');fig.tight_layout()
    output['rgb']=save_plot(fig,'frame153_rgb');output['crop']=list(box)
    fig,axes=plt.subplots(2,4,figsize=(10.5,5.4))
    frame=read(ROOT/'data/derived/kinematic_resplit/final_test/transforms_test.json')['frames'][index]
    with Image.open(ROOT/'data/derived/kinematic_resplit/final_test'/(frame['file_path']+'.png')) as im:
        target=im.getchannel('A').resize(gt.size,Image.Resampling.BICUBIC)
    panels=[np.asarray(target)]+[np.asarray(Image.open(m['directory']/'test/ours_50000/alpha'/name).convert('L')) for m in models]
    for ax,a,title in zip(axes.flat,panels,['GT alpha']+[m['id']+' alpha' for m in models]):
        ax.imshow(a,cmap='gray',vmin=0,vmax=255);ax.set_title(title);ax.axis('off')
    axes.flat[-1].axis('off');fig.tight_layout();output['alpha']=save_plot(fig,'frame153_alpha')
    return output


class Page(L.Page):
    def __init__(self,n,title,subtitle=''):
        RECORD.extend(['','## '+title,'',subtitle,''])
        super().__init__(n,'C0-C5 覆盖与形变消融',title,subtitle)
    def text(self,text,style='body',gap=10):
        RECORD.append(re.sub('<[^>]+>','',text.replace('<br/>','\n'))+'\n')
        super().text(text,style,gap)
    def heading(self,text):
        RECORD.extend(['','### '+text,'']);super().heading(text)
    def table(self,data,widths,**kwargs):
        RECORD.extend(['| '+' | '.join(map(str,row))+' |' for row in data[:1]])
        RECORD.append('| '+' | '.join(['---']*len(data[0]))+' |')
        RECORD.extend('| '+' | '.join(map(str,row))+' |' for row in data[1:]);RECORD.append('')
        super().table(data,widths,**kwargs)
    def image(self,path,height,caption=None):
        RECORD.extend(['',f'![{caption or Path(path).stem}](coverage_assets/{Path(path).name})',''])
        super().image(path,height,caption)
    def path(self,path):
        self.text(L.path_markup(str(path.resolve())),'path',gap=7)


def build(models,a,f):
    for name,file in [('CN','msyh.ttc'),('CN-Bold','msyhbd.ttc')]:pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file,subfontIndex=0))
    pdfmetrics.registerFontFamily('CN',normal='CN',bold='CN-Bold',italic='CN',boldItalic='CN-Bold')
    for name,file in [('Arial','arial.ttf'),('PathFont','consola.ttf')]:pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
    L.TOTAL_PAGES=8;L.page_records.clear()
    L.c=canvas.Canvas(str(PDF),pagesize=(L.W,L.H),pageCompression=1)
    L.c.setTitle('PIDG 覆盖与邻域消融实验报告 C0-C5');L.c.setAuthor('PhysGauss / PIDG Extension Project')
    cw=L.CW
    p=Page(1,'覆盖更准，为何图像反而变差？','C0-C5 已完成训练结果与代码机制分析 | 2026-10-27')
    p.callout('结论：本轮没有出现全面优于 V3 的新增方案。Mask 提高轮廓 IoU，却牺牲 RGB 重建；邻域项在有 Mask 时弥补部分退化；单独增密最接近基准，但收益小且有归因混杂。')
    p.heading('六组究竟改变了什么')
    p.table([['组别','Mask L1','局部仿射约束','定向增密'],['C0 / V3','0','0','关闭'],['C1','0.1','0','关闭'],
        ['C2','0.1','0.0001','关闭'],['C3','0.1','0.0001','开启'],['C4','0','0.0001','关闭'],['C5','0','0','开启']], [90,100,145,cw-335])
    p.heading('统一比较条件')
    p.text('Rubber Duck；相同 W256_D0 网络、V3 RGB 损失与共同时间轴。每组从完全相同的随机 step0 状态训练 50k，C0 复用已完成的对照。Iso 和速度一致性均关闭。120 训练 /30 验证 /50 测试，图像与时刻互不重叠；测试为交错未见时刻插值。')
    p.text('主表统一使用最终 50k 快照，四个 best 仅按验证集选择，不根据测试指标改选。正式输出为 1600x900 PNG。新增项主要是训练损失和点分配策略，未扩大 MLP；增密会改变 Gaussian 数量。')
    p.text('重要边界：场景有多只鸭子及障碍物。C2/C3/C4 未接入真实物体 ID 隔离，属于未分物体的空间邻域消融，不能声称已避免跨物体连接。','small')
    p.finish()

    p=Page(2,'最终 RGB 指标：基准 PSNR 最高','固定50张测试图；PSNR/SSIM/MS-SSIM 越高越好，LPIPS 越低越好')
    p.table([['组别','PSNR','SSIM','LPIPS','MS-SSIM'],*[[m['id'],f"{m['scores']['PSNR']:.4f}",*[f"{m['scores'][k]:.6f}" for k in METRICS[1:]]] for m in models]], [48,98,120,120,cw-386])
    p.heading('独立作用与条件作用要分开看')
    rows=[['比较','含义','PSNR 差 /dB','改善帧 /50']]
    for key,label in [('C1-C0','仅 Mask'),('C4-C0','仅邻域'),('C5-C0','仅增密'),('C2-C1','Mask 后加邻域'),('C3-C2','再叠加增密')]:
        c=a['comparisons'][key];rows.append([key,label,f"{c['metric_delta']['PSNR']:+.4f}",str(c['per_frame']['PSNR']['improved_frames'])])
    p.table(rows,[80,160,130,cw-370])
    p.text('C5 的 SSIM 和 LPIPS 略好于 C0，但 PSNR 与 MS-SSIM 略差，因此是局部指标权衡。C2 相对 C1 改善约 1.41 dB，仍比 C0 低约 2.57 dB，不能把“弥补退化”称为超过基准。')
    p.text('C3 虽在25/50帧上优于C2，平均仍下降1.29 dB。原始169-197的最后8个测试时刻平均下降5.57 dB，贡献了约69%的总体均值退化，说明失败集中在部分时段。','small')
    p.text('这些是单次运行的观测差异。后文发现，在新增项真正启用之前曲线已分岔，因此表中差值不能直接当成严格因果增益。','small')
    p.finish()

    p=Page(3,'Alpha 指标揭示另一种权衡','轮廓、内部覆盖与 RGB 质量不是同一个目标')
    rows=[['组别','前景 PSNR','Mask IoU','内部低覆盖 %','背景误覆盖 %']]
    for m in models:
        q=m['coverage']['mean'];diag=m['alpha_diagnostics']['mean']
        rows.append([m['id'],f"{q['foreground_psnr']:.3f}",f"{q['mask_iou']:.4f}",f"{100*q['interior_hole_rate']:.5f}",f"{100*diag['background_false_positive_rate']:.2f}"])
    p.table(rows,[44,105,95,133,cw-377])
    p.image(f['alpha'],232,'原始153帧的完整 alpha。白色表示高不透明度；显示范围统一为0-1，未按模型单独归一化。')
    p.text('C0 的低内部空洞率伴随很低的轮廓 IoU。补充诊断显示它在 GT 背景也保留大量高 alpha：背景变黑不等于背景透明。Mask 能去掉这类误覆盖，但并不自动保证鸭子内部颜色、位置与遮挡顺序正确。')
    p.text('低覆盖率只检测透明缺口。高 alpha 的黑色像素也可能看起来像“洞”，但由颜色或遮挡错误形成的这类缺陷不会被计入低覆盖率，仍需检查前景 RGB 误差。','small')
    p.text('定义：IoU 使用 alpha>0.5；内部低覆盖率在 GT alpha>0.95 的3像素腐蚀区域统计预测 alpha<0.5；背景误覆盖率在 GT alpha<=0.5 区域统计预测 alpha>0.5。前景 PSNR 使用 GT alpha>0.5。均为逐帧平均。','small')
    p.finish()

    p=Page(4,'曲线审计：增密前已存在差异','共同随机初始化并不等于 GPU 训练全过程逐位一致')
    p.image(f['curves'],182,'每1k验证一次。后期验证指标改善趋缓，模型落在不同重建水平；只有验证曲线，不能据此断言过拟合。')
    p.image(f['prefix'],166,'阴影区域为首次补点前。覆盖证据从16k采样，首次实际新增 Gaussian 在18k，且18k验证发生在补点之前。')
    prefix=a['pre_activation_validation_psnr_differences']
    p.table([['验证 PSNR 差','15k：补点前','50k：最终'],
        ['C5-C0',f"{prefix['C5-C0']['15000']:+.4f}",f"{prefix['C5-C0']['50000']:+.4f}"],
        ['C3-C2',f"{prefix['C3-C2']['15000']:+.4f}",f"{prefix['C3-C2']['50000']:+.4f}"]], [170,170,cw-340])
    p.text('C3 在补点前就落后 C2 约0.80 dB；C5 在补点前反而领先 C0 约0.54 dB。因此最终差异不全是补点造成。具体非确定性来源尚未定位，自定义 CUDA 累加与增密阈值放大是候选因素。','small')
    p.text('邻域项也有同类限制：3k首次启用前，C4-C0 和 C2-C1 已分别相差约-0.08和+0.07 dB。更可靠的归因需要共同状态分叉及多 seed 重复。','small')
    p.finish()

    p=Page(5,'同一帧的 RGB 与逐帧表现','原始帧153 /导出00038；沿用此前讨论的困难帧，统一 GT 定义裁剪')
    p.image(f['rgb'],296,'GT 和 C0-C5 使用相同裁剪坐标。最后一格是 C1 的逐像素 RGB 平均绝对误差，色标范围固定0-0.25；该帧仅作问题定位。')
    p.image(f['frames'],172,'全部50个测试时刻的 PSNR；横轴为原始帧号，线段不代表计算了训练/验证帧的测试指标。')
    p.text('该帧 C1 出现明显颜色偏差，C2/C3 仍有局部形状或遮挡误差。可见“alpha 接近 GT”不保证颜色也正确。场景的多物体结构同时说明，只有距离门控不能证明邻域属于同一个物体。','small')
    p.text('蓝色障碍物上的黑色缺口提醒我们：视觉上的“洞”可能是透明缺口，也可能是高 alpha 的错误颜色或遮挡，需与上一页的 alpha 图一起判断。','small')
    p.finish()

    p=Page(6,'为什么会这样：证据与推断分开','下面是与当前观测一致的解释，不是已验证的唯一原因')
    p.heading('1. Mask 改变了可用的重建方式')
    p.text('证据：C1 的轮廓 IoU 从 C0 的0.2064提高到0.9764，但全图 PSNR 下降约3.98 dB，前景 PSNR 也下降约4.23 dB，说明退化并非只发生在背景。')
    p.text('机制解释：黑背景下，RGB 合成允许高 alpha、近黑颜色的 Gaussian 仍呈现黑色。额外 alpha 监督会压制这种解，并同时改变尺度、透明度、位置和形变的梯度。它更接近轮廓目标，却可能打乱原本有利于 RGB 的分配。Mask 权重0.1从第1步作用于全图，背景约占90%；这是值得检验的梯度权衡，但没有权重扫描，不能断言0.1必然过大。')
    p.heading('2. 邻域项既能稳定，也会限制表达')
    p.text('证据：C2 相对 C1 改善 RGB 和内部覆盖；C4 单独加邻域却低于 C0。候选解释是：局部协同能约束 Mask 条件下的不协调运动，但也会抑制需要更复杂非仿射变化的局部细节。场景有多个物体，而当前 canonical 欧氏近邻没有身份隔离，还可能错误耦合接触物体；本轮没有标签证实实际跨物体边数。')
    p.heading('3. 多一些点并不自动补出正确几何')
    p.text('证据：C3 实际加879点，C5加717点，最终分别为70,258和70,848点；C0为75,542点。常规增密阶段已经造成点数差异，所以“定向增密组”最终未必比基准更多。策略只复制弱覆盖像素已有贡献者附近的点；无贡献者的真正空白区无法确定深度，继承颜色/尺度/轨迹也可能延续错误。')
    p.text('C3 相对 C2 退化不能全部归因于增密；启用前已有差距。C5 只在部分指标接近或略超 C0，尚无全面、稳定的重建增益证据。','small')
    p.finish()

    p=Page(7,'下一步：先修正验证方式，再调强度','建议顺序；本报告不启动任何新的训练')
    p.table([['优先级','可执行对照','解决什么问题'],
        ['1','为每个增密对照使用同一个15k完整状态分叉；保持总预算50k，再做至少3个seed。','消除补点前已分岔的混杂，确认约0.3 dB差异是否可复现。'],
        ['2','从随机初始化对照较弱 Mask 权重及延迟/ramp，记录 RGB/Mask 对相同参数的梯度范数。','检验轮廓约束强度和启用时机，而非直接归咎模型容量。'],
        ['3','已知物体ID/可靠邻接过滤，并在 Gaussian 克隆时传播ID；做边界和接触区域检查。','排除跨物体约束；不以颜色或距离直接宣称物体身份。'],
        ['4','对比 alpha 假阳性/假阴性、前景RGB与局部误差；单独记录新点贡献及实际覆盖改善。','防止低空洞率被大面积不透明背景“刷好”，检验补点是否有效。']], [45,270,cw-315])
    p.heading('目前可以保留什么')
    p.text('C0 继续作为 RGB 主基准；C5 可保留为轻量增密候选，但不设成默认改进。Mask 显著改善轮廓，应保留为可调监督项，而不是直接判定无用。邻域项应先解决邻接可靠性，再判断是否值得加入最终模型。')
    p.heading('结论的边界')
    p.text('只有一个场景和一个 seed，没有重复实验置信区间；Gaussian 数量是优化结果而非固定容量。当前是交错时刻插值，非长时间缺口或未来预测。测试集有历史和本轮用户授权的提前查看，不能视为完全未触碰测试集。没有 GT 深度、速度或力的恢复结论。','small')
    p.text('训练与验证使用严格分离的120/30帧，固定50测试帧只用于本报告的评价；四个 best 均来自验证选择。本报告主表统一 final50k，避免通过测试结果重新挑选快照。','small')
    p.finish()

    p=Page(8,'渲染文件保存在哪里','以下均为绝对地址；每组5种快照共250张拼接图，图库可切换快照和帧')
    for m,root in zip(models,ROOTS):
        p.heading(m['id']+' / '+m['name'])
        p.path(root/'render_comparisons'/m['name']/'index.html')
        p.path(m['directory']/'test/ours_50000/renders')
    p.text('GT 在 renders 的同级 gt 文件夹；alpha 在同级 alpha；最终50k拼接在图库目录的 pairs/50000。其他快照为 best、metric_ssim、metric_lpips、metric_ms_ssim。','small')
    p.text('来源：各模型 results.json、per_view.json、coverage_metrics_test.json、metric_history.jsonl、metric_best.json、配置、初始化与渲染校验记录，以及当前 runtime/model/density 源码。完整 SHA-256 与逐帧诊断见配套 sources.json 和 assessment.json。','small')
    p.finish();L.c.save()


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    metadata,models=load();alpha_diagnostics(metadata,models);a=analyze(models);f=figures(models);build(models,a,f)
    MD.write_text('# PIDG 覆盖与邻域消融实验报告\n'+'\n'.join(RECORD),encoding='utf-8')
    ASSESS.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for p in [Path(__file__),ROOT/'variants/coverage_tuning/runtime.py',ROOT/'variants/coverage_tuning/model.py',ROOT/'variants/coverage_tuning/density.py',ROOT/'variants/coverage_isolated/runtime.py']:
        SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    proof={'pdf':str(PDF),'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),'expected_pages':8,
        'pages':L.page_records,'sources':SOURCES,'frame153_shared_crop':f['crop'],'data_checks':a['protocol']}
    PDF.with_suffix('.sources.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'pdf':str(PDF),'pages':L.page_records,'alpha_means':{m['id']:m['alpha_diagnostics']['mean'] for m in models}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
