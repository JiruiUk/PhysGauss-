"""Chinese method and experimental report, generated only from local artifacts."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
import pidg_ablation_report_layout as L

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/pdf'
TMP = ROOT/'tmp/pdfs/kinematic_report'
QA = OUT/'qa_kinematic_report'
PDF = OUT/'PIDG_Kinematic_Method_Report_CN.pdf'
RUNS = ROOT/'runs/pidg_kinematic_tuning'
EXP = ROOT/'experiments/kinematic_tuning'
NAMES = ['V1_shared_time', 'V2_shared_time_kinematic', 'V3_frame_time', 'V4_frame_time_kinematic']
METRICS = ['PSNR', 'SSIM', 'LPIPS', 'MS-SSIM']
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
records, sources, figure_checks = [], {}, []


def load():
    assessment = read(EXP/'assessment.json')
    assert read(RUNS/'controller.json')['status'] == 'completed'
    result = {}
    for name in NAMES:
        directory = RUNS/name/'rubber_duck'
        for relative, expected in assessment['models'][name]['source_sha256'].items():
            assert digest(ROOT/relative) == expected, relative
        assert read(directory/'checkpoints/resume.json')['iteration'] == 50000
        scores = read(directory/'results.json')['ours_50000']
        per_view = read(directory/'per_view.json')['ours_50000']
        assert all(set(per_view[k]) == {f'{i:05d}.png' for i in range(50)} for k in METRICS)
        assert all(abs(math.fsum(per_view[k].values())/50-scores[k]) < 1e-5 for k in METRICS)
        assert read(RUNS/'render_comparisons'/name/'manifest.json')['pair_count'] == 250
        result[name] = {'directory': directory, 'scores': scores, 'per_view': per_view,
                        'history': [json.loads(s) for s in (directory/'metric_history.jsonl').read_text().splitlines()],
                        'diagnostics': [json.loads(s) for s in (directory/'kinematic_diagnostics.jsonl').read_text().splitlines()]}
    source_files = [EXP/'assessment.json', EXP/'assessment.md', EXP/'initial_timing_assessment.json',
                    EXP/'source_freeze.json', EXP/'preflight.json', EXP/'derivative_audit.json',
                    ROOT/'variants/kinematic_tuning/runtime.py', ROOT/'variants/kinematic_tuning/model.py',
                    ROOT/'variants/kinematic_tuning/timeline.py', ROOT/'variants/kinematic_tuning/run.py',
                    ROOT/'scripts/build_kinematic_report.py', ROOT/'PIDG_REPO_MAP.md', ROOT/'data/PIDG/rubber_duck/transforms_train.json',
                    ROOT/'data/PIDG/rubber_duck/transforms_test.json']
    sources.update({str(p.relative_to(ROOT)): digest(p) for p in source_files})
    return assessment, result


def chart_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#253649', 'text.color': '#253649',
                         'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})


def save(fig, name):
    path = TMP/(name+'.png')
    fig.savefig(path, dpi=230, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def figures(data):
    chart_style(); out = {}
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7))
    for row, pair in enumerate([NAMES[:2], NAMES[2:]]):
        for col, key in enumerate(['PSNR', 'LPIPS']):
            ax = axes[row, col]
            for name, color in zip(pair, ['#3c789e', '#087f8c']):
                h = data[name]['history']
                ax.plot([v['iteration']/1000 for v in h], [v['metrics'][key] for v in h],
                        label=name.split('_')[0], lw=1.6, color=color)
            ax.axvline(1, color='#b56a29', ls='--', lw=1)
            ax.set(xlabel='Additional training steps (k)', ylabel=key,
                   title=('Metadata time' if row == 0 else 'Frame-index time')+' / '+key)
            ax.grid(alpha=.15); ax.legend(frameon=False)
    fig.tight_layout(h_pad=2, w_pad=2); out['curves'] = save(fig, 'curves')

    a, b = [data[name]['per_view'] for name in NAMES[2:]]
    frames = sorted(a['PSNR']); delta = np.array([b['PSNR'][n]-a['PSNR'][n] for n in frames])
    fig, ax = plt.subplots(figsize=(10.6, 3.7))
    ax.bar(range(50), delta, color=['#087f8c' if v >= 0 else '#b56a29' for v in delta])
    ax.axhline(0, color='#637487', lw=.8); ax.axvspan(-.5, 7.5, color='#087f8c', alpha=.07)
    ax.set(xlabel='Official test image index (0-49)', ylabel='V4 - V3 PSNR (dB)',
           title='Average gain is concentrated in the first eight test images')
    ax.set_xticks(range(0, 50, 5)); ax.grid(axis='y', alpha=.15)
    fig.tight_layout(); out['per_view'] = save(fig, 'per_view')

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.7))
    for name, color in zip(NAMES[2:], ['#3c789e', '#087f8c']):
        h = data[name]['history']; early = [v for v in h if v['iteration'] <= 5000]
        axes[0].plot([v['iteration'] for v in early], [v['metrics']['PSNR'] for v in early], 'o-',
                     ms=4, lw=1.5, label=name.split('_')[0], color=color)
        ds = data[name]['diagnostics']; means = [statistics.mean(t['relative_rmse'] for t in v['times']) for v in ds]
        axes[1].plot([v['iteration']/1000 for v in ds], means, color=color, label=name.split('_')[0])
    axes[0].axvline(1000, color='#b56a29', ls='--', label='Loss starts after 1k')
    axes[0].set(xlabel='Additional training steps', ylabel='PSNR (dB)', title='Advantage precedes the new loss')
    axes[1].axhline(1, color='#637487', ls='--', lw=.8)
    axes[1].set(xlabel='Additional training steps (k)', ylabel='Mean relative velocity RMSE', title='Self-consistency remains weak')
    for ax in axes: ax.grid(alpha=.15); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(w_pad=2); out['attribution'] = save(fig, 'attribution')

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    for name, color in zip(NAMES[2:], ['#3c789e', '#087f8c']):
        points = data[name]['diagnostics'][-1]['times']; t = [p['time'] for p in points]
        axes[0].plot(t, [p['trajectory_rms'] for p in points], 'o-', color=color, label=name[:2]+' trajectory')
        axes[0].plot(t, [p['predicted_rms'] for p in points], 's--', color=color, label=name[:2]+' predicted v')
        axes[1].plot(t, [p['relative_rmse'] for p in points], 'o-', color=color, label=name[:2])
    axes[0].set(xlabel='Normalized training time', ylabel='RMS speed (scene units / time)', title='Predicted speed and trajectory derivative')
    axes[1].axhline(1, color='#637487', ls='--', lw=.8)
    axes[1].set(xlabel='Normalized training time', ylabel='Relative velocity RMSE', title='Five fixed training-time probes')
    for ax in axes: ax.grid(alpha=.15); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(w_pad=2); out['physics'] = save(fig, 'physics')

    image_paths = [data[NAMES[2]]['directory']/'test/ours_50000/gt/00004.png']
    image_paths += [data[n]['directory']/'test/ours_50000/renders/00004.png' for n in NAMES[2:]]
    images = [np.asarray(Image.open(p).convert('RGB')) for p in image_paths]
    other_gt = np.asarray(Image.open(data[NAMES[3]]['directory']/'test/ours_50000/gt/00004.png').convert('RGB'))
    assert np.array_equal(images[0], other_gt)
    yy, xx = np.where(images[0].max(axis=2)>15)
    roi = (max(0,int(xx.min())-20), max(0,int(yy.min())-20), min(1600,int(xx.max())+21), min(900,int(yy.max())+21))
    yellow = (images[0][:,:,0] > 100)&(images[0][:,:,1] > 65)&(images[0][:,:,2] < .75*images[0][:,:,0])
    labels, count = label(yellow)
    objects = [(i+1, box) for i, box in enumerate(find_objects(labels)) if box is not None and np.count_nonzero(labels[box] == i+1)>100]
    _, box = min(objects, key=lambda item:item[1][0].start)
    zoom = (max(0,box[1].start-30), max(0,box[0].start-25), min(1600,box[1].stop+30), min(900,box[0].stop+25))
    for name, crop, size in [('visual', roi, (10.6,5)), ('zoom', zoom, (10.6,2.2))]:
        fig, axes = plt.subplots(1, 3, figsize=size)
        for ax, arr, title in zip(axes, images, ['GT', 'V3 / 28.9700 dB', 'V4 / 39.4435 dB']):
            ax.imshow(arr[crop[1]:crop[3],crop[0]:crop[2]]); ax.set_title(title, fontsize=11); ax.axis('off')
        fig.tight_layout(w_pad=.7); out[name] = save(fig, name)
    figure_checks.append({'test_index':4, 'gt_identity_verified':True, 'scene_roi_xyxy':roi,
                          'top_object_zoom_xyxy':zoom, 'note':'Identical GT-defined crops; display scaling only; original pixels retained in run artifacts',
                          'sources':{str(p):digest(p) for p in image_paths}})
    return out


class Page(L.Page):
    def __init__(self, number, title, subtitle=''):
        super().__init__(number, '方法分析与完整实验复盘', title, subtitle)
        L.c.bookmarkPage('section_'+str(number))
        L.c.addOutlineEntry(str(number)+'. '+title, 'section_'+str(number), level=0)
        records.append('\n## '+str(number)+'. '+title+'\n'+subtitle+'\n')
    def text(self, text, style='body', gap=10):
        records.append(text+'\n'); super().text(text, style, gap)
    def heading(self, text):
        records.append('\n### '+text+'\n'); super().heading(text)
    def table(self, data, widths, highlight=None, size='table'):
        records.extend([' | '.join(str(v) for v in row) for row in data]);records.append('')
        super().table(data, widths, highlight, size)
    def callout(self, text):
        records.append('> '+text+'\n'); super().callout(text)
    def finish(self):
        c = L.c; c.setStrokeColor(colors.HexColor('#D4E0E8')); c.line(L.LEFT,44,L.W-L.RIGHT,44)
        c.setFont('CN',7.3);c.setFillColor(L.GRAY)
        c.drawString(L.LEFT,30,'Rubber Duck | 50k 追加训练 | 固定 50 张测试图 | 2026-10-25')
        c.setFont('Arial',8);c.drawRightString(L.W-L.RIGHT,30,f'{self.number:02d} / 15')
        L.page_records.append({'page':self.number,'title':self.title,'content_bottom':round(self.y,2)})
        c.showPage()
    def path(self, value):
        self.text(L.path_markup(str(value)), 'path', gap=7)


def architecture(page):
    c=L.c; x=L.LEFT; y=page.y; widths=[110,154,211]; gap=16; height=110
    page.check(height+22)
    texts=[('共同输入','canonical 坐标 X<br/>归一化时间 tau<br/>粒子 ID（速度分支）'),
           ('形变 / 渲染分支','Hash features + 门控 MLP<br/>Phi = R(q)X + b<br/>RGB 图像损失'),
           ('速度 / 应力分支','6 平面特征 + 时间编码<br/>MLP 43 → 64 → 64<br/>速度 3D / 应力 6D')]
    for w,(title,body) in zip(widths,texts):
        c.setFillColor(L.PALE);c.roundRect(x,y-height,w,height,5,fill=1,stroke=0)
        obj=L.P(title,'body');_,h=obj.wrap(w-16,100);obj.drawOn(c,x+8,y-12-h)
        obj=L.P(body,'small');_,h=obj.wrap(w-16,100);obj.drawOn(c,x+8,y-38-h)
        x+=w+gap
    page.y-=height+18
    page.callout('新增连接：从形变分支计算轨迹时间导数，与速度分支输出做一致性损失；梯度同时回传到两条分支。网络宽度、深度和输出头不变。')


def build(a, data, f):
    for name,file in [('CN','msyh.ttc'),('CN-Bold','msyhbd.ttc')]:
        pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file,subfontIndex=0))
    pdfmetrics.registerFontFamily('CN',normal='CN',bold='CN-Bold',italic='CN',boldItalic='CN-Bold')
    for name,file in [('Arial','arial.ttf'),('PathFont','consola.ttf')]:pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
    L.c=canvas.Canvas(str(PDF),pagesize=(L.W,L.H),pageCompression=1)
    L.c.setTitle('PIDG 时间尺度统一与速度-轨迹导数一致性：方法、实验与启发')
    L.c.setAuthor('PhysGauss / PIDG Extension Project');L.page_records.clear()
    cw=L.CW

    p=Page(1,'时间尺度与速度-轨迹一致性','PIDG 方法动机、代码差异、完整长训练结果与后续研究启发')
    p.text('研究问题：渲染出的运动轨迹与物理网络预测的速度，是否描述同一个运动？本报告以本仓库实际实现和已保存实验产物为依据，复盘统一时间及新增运动学一致性约束的动机、实现、结果和证据边界。')
    p.callout('<b>结论：</b>V4 是本次运行中平均重建指标最好的候选，但现有实验尚不能把优势稳定归因于新约束，速度场也尚未与轨迹导数有效对齐。')
    p.table([['已完成实验','训练与评估','可视化产物'],['V1-V4 共 4 组','每组追加 50k；固定 50 张测试图','5 个快照标签 × 50 帧 × 4 组'],['单场景 / seed 0','1600 × 900；最终 PNG 指标核验','共 1000 张 GT / Render 拼接图']], [125,185,cw-310])
    p.heading('读者应带走的四点')
    p.text('1. 原实现的图像形变与速度场存在耦合缺口，且物理时间输入多除了一次训练帧数。<br/>2. 共同时轴能显著改变评分，但这同时改变评测输入，必须与模型改进分开归因。<br/>3. V4 的平均 PSNR 优势集中在早期少数漏物体帧；大均值增益不等于全帧改善。<br/>4. 在新损失启用前，对照组已经出现明显差异；下一轮应从同一完整 checkpoint 分叉。')
    p.heading('报告结构')
    p.text('第 2-5 页说明动机、原结构、时间与新目标；第 6-12 页给出实验设置、曲线、逐帧证据、物理诊断及归因检查；第 13 页提出后续验证路线；第 14-15 页列出绝对产物路径和可追溯来源。','small')
    p.finish()

    p=Page(2,'为什么需要这项约束？','把“图像中在运动”与“物理网络认为在运动”联系起来')
    p.heading('原方法中有两条不同的学习路径')
    architecture(p)
    p.text('渲染分支通过图像误差优化 Gaussian 及其形变；速度/应力网络通过动量残差学习场量。本场景 use_flow=False，原有光流轨迹损失未启用。两条分支会通过共享坐标间接关联，但没有强制要求预测速度等于渲染轨迹的时间导数。')
    p.heading('缺口为何会影响可解释性')
    p.text('如果速度近似常量、应力变化很小，动量残差可能很低；与此同时，形变网络仍可通过时间标签生成明显运动。低物理残差或高 PSNR 因而都不能单独证明预测速度对应图像运动。运动学一致性是一项必要联系，用于减少这种分支间不一致；它本身并不足以保证材料、应力或外力正确。')
    p.heading('本次没有扩大网络')
    p.text('形变仍采用 W256_D0：空间特征经过 Linear(32,256) 和方向门控；时间平面特征经过 Linear(192,256)+ReLU，两者逐元素相乘，再进入 7/4/3 通道输出头。depth=0 表示共享隐藏堆栈为空，并不表示整个网络没有线性层或非线性。速度网络仍是 43→64→64 的共享 MLP，接 3 维速度头和 6 维应力头。')
    p.text('本轮不启用 ISO、残差适配器或新的本构分支。比较重点是时间输入和损失连接，非网络容量消融。源码依据：[S1]-[S3]。','small');p.finish()

    p=Page(3,'时间尺度：修正什么，假设什么','共同物理变量不要求使用相同的内部编码坐标')
    p.table([['位置','原实现','本次开关启用后'],['形变 / 渲染','直接使用 fid','使用共同 tau'],['动量约束速度网络','fid / 150','使用同一 tau，移除额外 /150'],['训练 / 测试 metadata','分别归一化到 [0,1]','保留原协议，或单独使用帧编号协议']], [112,175,cw-287])
    p.text('原数据的 150 张训练图时间为 j/149，50 张测试图时间为 k/49；图像文件编号则共同覆盖 0001-0200。可选 frame_index 协议令 tau=(frame_id-1)/199，不改变 RGB、相机或 150/50 划分，也不改写原 JSON。')
    p.table([['图像编号','所属划分','原标签时间','共同帧时间'],['0001','测试','0','0'],['0002','训练','0','1/199 ≈ 0.005025'],['0005','测试','1/49 ≈ 0.020408','4/199 ≈ 0.020101'],['0200','训练','1','1']], [90,90,160,cw-340])
    p.callout('帧编号协议依赖“数字文件名代表均匀采样顺序”的假设。它不是已验证的真实时间戳。V1/V2 保留原标签；V3/V4 才使用该额外映射。')
    p.heading('归一化时间不等于真实秒数')
    p.text('若真实时间跨度为 T，且 t=t0+T·tau，则 v_phys=(1/T)·dPhi/dtau。当前未知 T，速度单位只能表述为“场景距离 / 单位归一化时间”。内部 hash 坐标缩放与 Fourier 时间编码仍保留；轨迹求导经过完整形变调用，包含这些编码的时间变化。')
    p.text('原 compute_acceleration 用自动微分求输入时间导数，传入的 dt 并未参与该导数；仅改 dt 数值不能修复上述额外除法。源码依据：[S2]-[S4]。','small');p.finish()

    p=Page(4,'新方法：让速度对应完整轨迹','对旋转和平移共同求导；保持原速度 / 应力预测分支')
    p.heading('运动学定义')
    p.callout('Phi(X,tau) = R(q(X,tau))·X + b(X,tau)<br/>v_traj(X,tau) = dPhi(X,tau) / dtau<br/>L_kin = mean Huber_delta(v_pred(X,tau) - v_traj(X,tau))')
    p.text('这里 X 是可优化的 canonical 参考坐标，不保证就是第 0 帧或无应变材料构型。7 通道 d_xyz 中，前 4 个通道是 quaternion，后 3 个通道是平移；它并非一个 3 维位移。R(q) 使用与渲染器相同的归一化 quaternion 旋转。')
    p.heading('为什么不能只对平移头求导')
    p.text('完整导数含有 dR/dtau·X + db/dtau。即便平移 b 不变，物体绕参考点旋转也会产生速度。只监督 db/dtau 会遗漏这部分运动。Gaussian 的协方差旋转/尺度输出是独立量，本项只对中心轨迹建立约束。')
    p.heading('梯度如何融合')
    p.text('预测速度和轨迹导数两侧都保留参数梯度。损失能更新时间/空间形变网格、形变输出网络和速度网络。采样的 X 被 detach，使该项不直接通过移动 canonical 坐标来减小误差；图像损失仍沿原路径更新 canonical Gaussian。')
    p.heading('它与原物理残差的关系')
    p.text('新项补充的是运动学联系，不替换 PIDG 的预测应力，也不引入新的自由刚度或逐 Gaussian 外力。原 canonical 坐标下的动量残差、convective contraction 和采样权重保持未改；因此“同一时间 + 速度一致性”仍不是完整物理正确性的证明。')
    p.heading('与原实现的差异清单')
    p.table([['项目','原始 PIDG','本轮方法'],['网络 / 参数量','形变、速度、应力分支','结构不变；未新增可学习参数'],['训练目标','图像 + 特征正则 + 原物理项','T3 感知项 + 可选速度一致性'],['推理路径','R(q)X+b 渲染','渲染器不变；可选重映射时间'],['监督来源','训练 RGB 等现有输入','仍无 GT 速度、力或新物理标签']], [86,183,cw-269])
    p.text('源码依据：[S1]-[S3]。新项是训练期正则，图像推理不需额外计算轨迹导数；全部新功能可关闭。','small');p.finish()

    p=Page(5,'数值实现与训练目标','差分可反传；工程可用性与方法有效性分开验证')
    p.heading('为什么使用可微有限差分')
    p.text('为了把轨迹时间导数的损失反传进模型，直接使用导数图通常涉及高阶或混合导数。当前自定义 CUDA hash encoder 的高阶输入导数支持有限。本轮采用普通参数反传可处理的轨迹差分，避免依赖未覆盖的坐标二阶导数。')
    p.callout('内部时刻：v_traj ≈ [Phi(tau+h)-Phi(tau-h)] / (2h)<br/>起点附近：[-3Phi(tau)+4Phi(tau+h)-Phi(tau+2h)] / (2h)<br/>终点附近：[3Phi(tau)-4Phi(tau-h)+Phi(tau-2h)] / (2h)<br/>h = 0.001；二阶精度差分；查询均位于 [0,1]。')
    p.table([['配置','取值 / 行为'],['Huber delta / lambda','0.1 / 0.001'],['采样 / 间隔','256 个粒子；每 10 步；确定性偏移采样'],['启用 / 渐增','iter >1000；1000 步 ramp；首次可能施加于 1010'],['稀疏施加补偿','执行步的系数为 lambda × ramp × 10；其余步为 0'],['差分查询','暂关随机特征正则；清空 attention 缓存；查询后恢复']], [132,cw-132])
    p.text('L_total = L_T3 + lambda·ramp·10·L_kin（仅采样迭代施加）。Huber 小误差为 0.5e²，大误差为 delta·(|e|-0.5delta)。间隔补偿表达的是平均目标权重，不等价于每步施加时完全相同的优化轨迹。')
    p.heading('为何采用保守权重与 Huber')
    p.text('轨迹导数在快速运动或局部变形处可能出现大残差。Huber 在大误差区的梯度受 delta 限制，配合小权重和 ramp，意在减小新项对已有图像重建的冲击。代价是大误差样本的监督可能偏弱；两侧同时更新还允许通过改变轨迹来降低损失。本轮只测试一个权重，未证明该配置最优。','small')
    p.text('验证结果：38 项相关单元测试通过；真实 CUDA 网格/形变头/速度参数均有有限非零梯度；20→100 完整状态恢复与 PNG 指标核验通过。128 个冻结模型点上，h=0.001 与输入自动微分导数的相对 RMSE 为 0.815%，h=0.0005 时为 0.397%。这只验证局部导数近似，不是真实速度精度。','small');p.finish()

    p=Page(6,'实验设计与公平比较','将时间协议、训练预算、损失改变和快照选择分别记录')
    p.table([['变体','时间输入','一致性权重','比较目的'],['V0 / T3','原 metadata；物理保留旧缩放','0','复用已完成 T3'],['V1','原 metadata；物理共享时间','0','只修物理时间输入'],['V2','与 V1 相同','0.001','原协议下的损失对照'],['V3','共同帧编号时间','0','新时间协议控制组'],['V4','与 V3 相同','0.001','新协议下的损失对照']], [64,195,75,cw-334])
    p.text('四组都从原始 PIDG W256_D0 最佳 50k 权重出发，新实验初始化新优化器，再追加 50k；继续运行时恢复完整模型、优化器、RNG 和相机采样状态。单场景 Rubber Duck，seed 0，训练 150 图、测试 50 图。全程关闭 densification 和 ISO。')
    p.heading('共享的 T3 配方')
    p.text('图像目标为 0.8·L1 + 0.2·(1-SSIM)，加 VGG LPIPS 权重 0.02；感知项每 4 步在 800×450 图像计算并乘以 4。空间/时间特征正则权重各 0.5；原物理残差权重 0.01，每 100 步计算。图像渲染与正式评估仍用 1600×900。')
    p.table([['优化设置','共同值'],['位置学习率','2e-5 → 1e-7；50k 调度'],['形变网格 / 网络倍率','25 / 5'],['速度网络学习率','1e-4'],['评价 / 保存','每 1k 测试；保留四项最佳及最终 50k 快照']], [155,cw-155])
    p.text('本报告以最终 50k PNG 快照作主要比较。四指标最优快照按测试集选择，沿用历史流程；没有独立、未参与选择的验证集。原始 PIDG 只有 50k 训练，本轮总计约 100k；与原模型的差值还包含额外计算和 T3 损失影响。','small');p.finish()

    p=Page(7,'最终重建指标：分协议解读','正式 PNG 指标；每行同样覆盖固定的 50 张官方测试图')
    original=read(ROOT/'runs/pidg_official/rubber_duck/results.json')['ours_best']
    t3=read(ROOT/'runs/pidg_metric_tuning/T3_baseline_perceptual/rubber_duck/results.json')['ours_metric_lpips']
    def metricrow(label,s):return [label,f"{s['PSNR']:.4f}",*[f'{s[k]:.6f}' for k in METRICS[1:]]]
    widths=[132,80,94,94,cw-400]
    p.heading('原 metadata 时间协议')
    p.table([['模型','PSNR','SSIM','LPIPS','MS-SSIM'],metricrow('原 PIDG（仅 50k）',original),metricrow('T3（再训练 50k）',t3),*[metricrow(n[:2],data[n]['scores']) for n in NAMES[:2]]],widths)
    p.text('V2 相对原始 PIDG：PSNR +1.0128 dB、LPIPS 降低 5.84%。但 T3 已达到 28.2648 dB；V2 相对 T3 仅 +0.0048 dB，相对 V1 为 +0.0180 dB，且 SSIM / LPIPS 略降。没有稳定的综合优势。')
    p.heading('共同帧编号时间协议')
    p.table([['模型','PSNR','SSIM','LPIPS','MS-SSIM'],*[metricrow(n[:2],data[n]['scores']) for n in NAMES[2:]]],widths,highlight=2)
    p.text('V4 相对 V3：PSNR +1.4233 dB，SSIM +0.000163，LPIPS 降低 3.64%，MS-SSIM +0.000866。四项平均指标均改善。第 9-12 页检查这些改善是否广泛、是否能归因于一致性项。')
    p.callout('同一原始权重，不训练而仅改测试时间：27.2568 → 33.2223 dB（+5.9655）。时间输入变化本身即可大幅改变评分。因此 V3/V4 与历史 27 dB 的跨协议差不能解释为结构或损失增益。')
    p.finish()

    p=Page(8,'训练曲线：收敛与收益时段','每 1k 的 PNG 等价评估；最终快照另经真实 PNG 核验')
    p.image(f['curves'],337,'图 1. 上排为原时间协议，下排为共同帧时间；纵轴范围分别设置。虚线为第 1k 步，新增损失只会在此后启用。')
    p.text('V1/V2 的最终 PSNR 与 LPIPS 非常接近，SSIM / MS-SSIM 最优点落在 10k 左右；继续训练主要继续降低 LPIPS。V3/V4 的四项最优点都落在 50k，后期仍有小幅改善，说明它们的最终指标并非仅取自短训尖峰。')
    p.text('V4 在第 1000 步已经领先 V3：36.1283 vs 35.2620 dB；第 10k 为 37.9050 vs 36.5991 dB；第 50k 为 38.7692 vs 37.3459 dB。领先从损失启用前便已存在，这是比最终均值更重要的归因检查点。')
    p.callout('这些是单场景、单 seed 的一次运行轨迹，没有重复试验置信区间。曲线平滑或长期领先，都不能单独排除执行非确定性与早期训练路径差异。');p.finish()

    p=Page(9,'逐帧分析：均值没有说明全部','V4 相对 V3；按官方导出文件名 00000-00049 排序')
    p.image(f['per_view'],184,'图 2. 正值为 V4 的 PSNR 更高。前 8 张贡献明显，后续多数差值接近 0 或略为负值。')
    pair=a['paired_comparisons'][1]
    p.table([['指标','改善图像数 / 50','逐帧差中位数'],*[ [k,f"{pair['per_view'][k]['improved_views']} / 50",f"{pair['per_view'][k]['median_delta']:+.6f}"] for k in METRICS]], [145,170,cw-315])
    p.text('V4 仅 23/50 张图的 PSNR 优于 V3；逐帧 PSNR 差的中位数为 -0.0215 dB。前 8 张对 50 张平均差贡献 +1.4406 dB；剩下 42 张平均差为 -0.0207 dB。平均 +1.4233 dB 主要反映少数明显失败帧的恢复，而非全帧同幅改善。')
    p.heading('PSNR 平均值与像素 MSE 不是同一种平均')
    p.text('每张归一化 RGB 图有 MSE_i=10^(-PSNR_i/10)。从逐帧分数还原并平均 MSE，V4 相对 V3 下降约 51.21%；直接把平均 PSNR 差换算为误差比，得到的是几何平均意义。报告同时保留逐帧分布，避免混淆这两个统计量。','small');p.finish()

    p=Page(10,'视觉证据：恢复一个漏掉的目标','官方测试图 00004；三列使用完全相同的 GT 定义裁剪区域')
    p.image(f['visual'],241,'图 3. GT、V3、V4 的同相机、同帧时间对照。图中显示的是主体裁剪和排版缩放；原分辨率图像保存在 runs。')
    p.image(f['zoom'],105,'图 4. 对同一最上方黄色目标区域放大。V3 未显示该鸭子；V4 恢复了目标，但细节仍偏模糊。')
    p.table([['测试帧','V3 PSNR','V4 PSNR','差值'],['00004.png','28.9700 dB','39.4435 dB','+10.4735 dB']], [130,125,125,cw-380])
    p.text('该样例支持“重建结果存在有意义的局部改善”。它不证明恢复来自速度一致性项，更不能由此判断速度、应力或外力正确。第 12 页给出损失启用前的对照差异。','small');p.finish()

    p=Page(11,'物理诊断：一致性尚未对齐','固定 256 个粒子、5 个训练时刻；不使用测试图 RGB 计算该诊断')
    p.image(f['physics'],190,'图 5. 左：轨迹导数与预测速度幅度；右：每个时刻的相对误差。速度单位为场景距离 / 归一化时间。')
    p.table([['最终诊断均值','V3','V4'],['速度与轨迹导数 RMSE','1.3832','1.2185'],['轨迹导数 RMS','1.3793','1.2155'],['预测速度 RMS','0.0533','0.1102'],['逐时刻相对 RMSE 的均值','1.0057','0.9914']], [235,130,cw-365])
    p.text('V4 的绝对误差下降 11.91%，同时轨迹导数幅度下降 11.87%。相对误差仍接近 100%，预测速度幅度明显偏低。结果与“部分通过降低轨迹导数减小损失”相容，但仅凭这些标量不能证明具体机制。')
    p.callout('工程上梯度已打通，不等于优化上已学到正确物理。当前没有速度/力真值；这些只能评价自一致性，不能称作真实速度误差或材料识别精度。');p.finish()

    p=Page(12,'归因审计：方法有效性尚待证明','新增损失之前已出现差异，必须调整下一轮对照设计')
    p.image(f['attribution'],188,'图 6. 左：早期 PSNR 轨迹；右：全程相对速度误差。第 1k 步对应的性能差不能来自尚未启用的一致性项。')
    p.table([['检查点','V3','V4','差值 / 含义'],['初始 PSNR','33.2223','33.2223','相同初始评分'],['第 1k PSNR','35.2620','36.1283','+0.8662，损失尚未启用'],['第 50k PSNR','37.3459','38.7692','+1.4233，归因存在混杂']], [115,105,105,cw-325])
    p.text('代码在 iteration≤1000 时返回，首次可能施加新项是 1010；V3/V4 的运行配置除一致性权重外相同。然而相同 seed、相同初始权重和相同初始评分，不保证数值执行过程完全一致。CUDA 原子梯度等非确定性或其他执行差异是待排查因素，本次并未定位具体原因。')
    p.text('不能把 1.4233-0.8662 简单视为“纯约束增益”：早期参数与优化器状态的差异会非线性地影响后续训练。原协议下 V1/V2 在第 1k 也存在差异，同样需要更严格的控制。')
    p.callout('可支持的结论：本次 V4 的平均重建指标最好。尚不支持的结论：该一致性项稳定造成 +1.42 dB 增益，或保证物理速度场正确。');p.finish()

    p=Page(13,'对后续研究与实现的启发','先消除归因混杂，再优化物理耦合，最后扩大实验范围')
    p.table([['优先级','建议实验 / 修改','需要回答的问题'],['P0','从同一第 1k 完整 checkpoint 分叉；复制优化器、RNG、相机顺序，只改 lambda','能否排除启用前的训练差异？'],['P0','先做无损失 / 无损失重复，再做至少 3 个 seed 的配对试验','无改动噪声多大？增益能否重复？'],['P0','核对模拟器/采集器真实时间戳；保留两套独立协议','共同帧时间假设是否正确？'],['P1','报告全图 + 前景 ROI + 分时段指标及失败帧','改善覆盖率如何？是否依赖少数漏物体帧？'],['P1','分支梯度范数、动态/静态采样与分阶段拟合对照','约束为何未对齐？是否出现运动幅度收缩？'],['P2','审计坐标变换及动量残差，再扩展物理解释','自一致性与守恒残差是否使用一致定义？']], [43,282,cw-325])
    p.heading('对损失与结构的具体启发')
    p.text('应在控制实验中比较：先冻结形变拟合速度，再以小权重联合优化；或考察带尺度记录的相对误差目标，防止绝对误差通过轨迹幅度下降而改善。相对目标的分母设计同样可能引入优化漏洞，需要梯度与退化解测试，不能直接当作已验证修复。')
    p.text('均匀采样可能让静态/低运动点主导。可利用已有训练前景或运动信息构建动态/静态分层采样对照，仍保留官方全图指标。这里不提出学习材料分割或增加逐 Gaussian 自由力。')
    p.text('本次没有改变 MLP 参数量，不能据此判断容量已足够或不足。时间语义、优化可复现性及两分支的耦合应优先审计；扩大网络前先确认当前网络确实收到并利用了正确监督。')
    p.text('以上均为后续建议，尚未运行。当前可在接受帧时间假设时保留 V4 用作重建展示；不将其作为物理识别成功的依据。','small');p.finish()

    p=Page(14,'渲染、GT 与拼接图的绝对地址','每组均覆盖 50 张官方测试图；以下列出最终 50k 快照')
    p.text('文件 00000.png-00049.png 按官方测试 JSON 顺序导出。网页可切换 best、metric_ssim、metric_lpips、metric_ms_ssim、50000 五个标签；相同迭代可能重复保存。每张拼接图左侧 GT、右侧 Render，原保存尺寸 3200×950（含 50 像素标题区），图像面板不缩放。','small')
    for name in NAMES:
        p.heading(name)
        base=RUNS/name/'rubber_duck/test/ours_50000'
        p.text('Render / GT：','small',gap=2);p.path(base/'renders');p.path(base/'gt')
        p.text('拼接图 / 网页：','small',gap=2)
        p.path(RUNS/'render_comparisons'/name/'pairs/50000')
        p.path(RUNS/'render_comparisons'/name/'index.html')
    p.finish()

    p=Page(15,'来源、复现与交付说明','所有数值来自本地已完成运行；报告未触发训练或修改模型')
    p.table([['编号','主要证据 / 实现位置'],['S1','PIDG_REPO_MAP.md；原始及独立 fork 的 network.py、renderer、motion_utils/pidg.py'],['S2','variants/kinematic_tuning/{model,runtime,timeline,prepare,run}.py'],['S3','experiments/kinematic_tuning/{plan,preflight,derivative_audit,source_freeze}.json'],['S4','data/PIDG/rubber_duck/transforms_{train,test}.json'],['S5','各变体 rubber_duck/{results,per_view,metric_best}.json 和 metric_history.jsonl'],['S6','各变体 kinematic_diagnostics.jsonl、kinematic_loss.jsonl、metric_config.json'],['S7','experiments/kinematic_tuning/assessment.{md,json}；initial_timing_assessment.json'],['S8','runs/pidg_official/rubber_duck/results.json；T3_baseline_perceptual/rubber_duck/results.json']], [38,cw-38])
    p.heading('复现与完整来源清单')
    p.text('生成命令：envs/pidg-cu128/python.exe scripts/build_kinematic_report.py。构建时核验 assessment 中记录的来源 SHA-256、每组最终 50k、50 个逐帧键，以及逐帧均值与 PNG 指标的一致性。配套 .sources.json 保存完整来源哈希和图像裁剪记录；.source.md 保留正文与表格文本。','small')
    p.path(OUT/'PIDG_Kinematic_Method_Report_CN.sources.json')
    p.path(EXP/'assessment.md')
    p.heading('边界与适用范围')
    p.text('本报告中的“原方法”指本仓库已审计的 PIDG 实现和历史训练流程，未将未核验的论文细节当作事实。仅一场景、一个 seed、一个一致性权重；官方测试图被用于历史式快照选择，尚无独立未触碰测试集。没有真实速度、外力或材料参数标签。','small')
    p.callout('最终建议：保留 V4 作为当前观测到的最佳重建候选；先完成共同 checkpoint 分叉、多 seed 和时间戳审计，再决定是否将速度一致性项设为默认，以及是否继续进行物理参数解释。')
    p.finish()
    L.c.save()


def main():
    for path in (OUT,TMP,QA):path.mkdir(parents=True,exist_ok=True)
    assessment,data=load();figs=figures(data);build(assessment,data,figs)
    (OUT/'PIDG_Kinematic_Method_Report_CN.source.md').write_text('\n'.join(records),encoding='utf-8')
    evidence={'created_utc':datetime.now(timezone.utc).isoformat(),'pdf_absolute':str(PDF),
              'pdf_sha256':digest(PDF),'sources':sources,'figure_pixel_checks':figure_checks,
              'pages':L.page_records,'expected_page_count':15,'all_final_png_metrics_verified':True}
    (OUT/'PIDG_Kinematic_Method_Report_CN.sources.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Created '+str(PDF),flush=True)
    print(json.dumps(L.page_records,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
