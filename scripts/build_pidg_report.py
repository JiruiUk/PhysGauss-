"""Build the Chinese report from saved metrics, code receipts and GT/render PNGs.

No training or model import is performed. Run with envs/pidg-cu128/python.exe.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
TMP = ROOT / 'tmp/pdfs/pidg_report'
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
PDF = OUT / 'PIDG_Variants_Report.pdf'
RENDER = ROOT / 'runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152'

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

results = read('experiments/results.json')
manifest = json.loads((RENDER / 'manifest.json').read_text(encoding='utf-8'))
rows = results['long_rows']
by_name = {r['configuration_id']: r for r in rows}
models = [r['configuration_id'] for r in rows]
best = {m['name']: m['best_snapshot_iteration'] for m in manifest['models']}
baseline = results['baseline']
M9 = read('experiments/M9_validation.json')

pdfmetrics.registerFont(TTFont('CN', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('CN-Bold', 'C:/Windows/Fonts/msyhbd.ttc', subfontIndex=0))
pdfmetrics.registerFontFamily('CN', normal='CN', bold='CN-Bold', italic='CN', boldItalic='CN-Bold')
pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
NAVY = colors.HexColor('#153149')
TEAL = colors.HexColor('#087F8C')
INK = colors.HexColor('#253649')
GRAY = colors.HexColor('#637487')
PALE = colors.HexColor('#EFF5F8')
ORANGE = colors.HexColor('#B56A29')
W, H = 595.276, 841.89
LEFT, RIGHT = 44, 44
CW = W - LEFT - RIGHT
styles = {}
for name, size, leading, color in [('body', 10.2, 17, INK), ('small', 8.5, 13, GRAY),
                                  ('caption', 8.5, 13, GRAY), ('table', 8.5, 13, INK),
                                  ('thead', 8.5, 13, colors.white)]:
    styles[name] = ParagraphStyle(name, fontName='CN', fontSize=size, leading=leading,
                                 textColor=color, wordWrap='CJK', alignment=TA_LEFT)

c = canvas.Canvas(str(PDF), pagesize=(W, H), pageCompression=1)
c.setTitle('PIDG 本构扩展：五个变体的结构与实验结果')
c.setAuthor('PhysGauss / PIDG Extension Project')
c.setSubject('Rubber Duck 50k 单场景实验、GT/Render 对照及条件可辨识性分析')
page_records = []

def P(text, style='body'):
    return Paragraph(text, styles[style])

class Page:
    def __init__(self, number, label, title, subtitle=''):
        self.number, self.label, self.title = number, label, title
        c.setFillColor(NAVY)
        c.rect(0, H - 11, W, 11, fill=1, stroke=0)
        c.setFont('Arial', 8)
        c.setFillColor(TEAL)
        c.drawString(LEFT, H - 39, 'PIDG / CONSTITUTIVE EXTENSIONS')
        c.setFont('CN', 8)
        c.setFillColor(GRAY)
        c.drawRightString(W - RIGHT, H - 39, label)
        c.setFont('CN-Bold', 22)
        c.setFillColor(NAVY)
        c.drawString(LEFT, H - 84, title)
        self.y = H - 104
        if subtitle:
            self.text(subtitle, 'small', gap=15)

    def check(self, height):
        if self.y - height < 58:
            raise RuntimeError(f'Page {self.number} overflow: y={self.y}, height={height}')

    def text(self, text, style='body', gap=10):
        obj = P(text, style)
        _, height = obj.wrap(CW, H)
        self.check(height + gap)
        obj.drawOn(c, LEFT, self.y - height)
        self.y -= height + gap

    def heading(self, text):
        self.check(29)
        c.setFillColor(TEAL)
        c.setFont('CN-Bold', 12)
        c.drawString(LEFT, self.y - 14, text)
        self.y -= 29

    def table(self, data, widths, highlight=None, size='table'):
        converted = [[P(escape(str(v)), 'thead' if i == 0 else size) for v in row]
                     for i, row in enumerate(data)]
        table = Table(converted, colWidths=widths, hAlign='LEFT')
        commands = [('BACKGROUND', (0, 0), (-1, 0), NAVY),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 7),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                    ('TOPPADDING', (0, 0), (-1, -1), 7),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                    ('LINEBELOW', (0, 0), (-1, 0), 0.4, NAVY)]
        for i in range(1, len(data)):
            commands += [('BACKGROUND', (0, i), (-1, i), PALE if i % 2 else colors.white),
                         ('LINEBELOW', (0, i), (-1, i), .3, colors.HexColor('#D8E3EA'))]
        if highlight:
            commands.append(('BACKGROUND', (0, highlight), (-1, highlight), colors.HexColor('#DDF0EC')))
        table.setStyle(TableStyle(commands))
        _, height = table.wrap(CW, H)
        self.check(height + 12)
        table.drawOn(c, LEFT, self.y - height)
        self.y -= height + 12

    def image(self, path, height, caption=None):
        self.check(height + 8)
        c.drawImage(str(path), LEFT, self.y - height, width=CW, height=height,
                    preserveAspectRatio=True, anchor='c', mask='auto')
        self.y -= height + 8
        if caption:
            self.text(caption, 'caption', gap=10)

    def callout(self, text):
        obj = P(text)
        _, height = obj.wrap(CW - 24, H)
        self.check(height + 31)
        c.setFillColor(PALE)
        c.roundRect(LEFT, self.y - height - 20, CW, height + 20, 5, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(LEFT, self.y - height - 20, 3, height + 20, fill=1, stroke=0)
        obj.drawOn(c, LEFT + 12, self.y - height - 10)
        self.y -= height + 31

    def finish(self):
        c.setStrokeColor(colors.HexColor('#D4E0E8'))
        c.setLineWidth(.5)
        c.line(LEFT, 44, W - RIGHT, 44)
        c.setFont('CN', 7.5)
        c.setFillColor(GRAY)
        c.drawString(LEFT, 30, 'Rubber Duck | 50k | 单场景、单随机种子 | 2026-10-19')
        c.setFont('Arial', 8)
        c.drawRightString(W - RIGHT, 30, f'{self.number:02d} / 13')
        page_records.append({'page': self.number, 'title': self.title, 'content_bottom': round(self.y, 2)})
        c.showPage()

def diagram():
    """Vector diagram. Main PIDG path is preserved; extension is an extra branch."""
    x0, y0, ww, hh = LEFT, 0, CW, 220
    def box(x, y, w, h, line1, line2='', active=False):
        c.setFillColor(colors.HexColor('#E0F0ED') if active else PALE)
        c.setStrokeColor(TEAL if active else colors.HexColor('#AABCCA'))
        c.roundRect(x0+x, y0+y, w, h, 4, fill=1, stroke=1)
        c.setFillColor(NAVY); c.setFont('CN-Bold', 9)
        c.drawCentredString(x0+x+w/2, y0+y+h/2+(3 if line2 else -3), line1)
        if line2:
            c.setFont('CN', 7.7); c.drawCentredString(x0+x+w/2, y0+y+h/2-12, line2)
    def arrow(x1, y1, x2, y2):
        c.setStrokeColor(GRAY); c.setFillColor(GRAY); c.setLineWidth(.8)
        c.line(x0+x1, y0+y1, x0+x2, y0+y2)
        angle=np.arctan2(y2-y1,x2-x1)
        path=c.beginPath();path.moveTo(x0+x2,y0+y2)
        for a in [angle+2.7,angle-2.7]:path.lineTo(x0+x2+5*np.cos(a),y0+y2+5*np.sin(a))
        path.close();c.drawPath(path,fill=1,stroke=0)
    box(0, 159, 106, 45, '规范高斯 X', '颜色、协方差、不透明度')
    box(134, 159, 105, 45, '形变场 D(X,t)', '旋转与平移等输出')
    box(268, 159, 105, 45, '变形高斯 x(t)', '沿用 PIDG 渲染状态')
    box(400, 159, 106, 45, '渲染图像', '与 GT 计算图像损失')
    arrow(106,181,134,181);arrow(239,181,268,181);arrow(373,181,400,181)
    box(0, 77, 106, 45, '速度 / 应力场', '保留 sigma_pred')
    box(134, 77, 105, 45, '局部拟合 F', 'X 与 x(t) 的邻域关系',True)
    box(268, 77, 105, 45, 'Neo-Hookean', '共享参数 mu、kappa',True)
    box(400, 77, 106, 45, '本构一致性', '预测应力与本构应力',True)
    arrow(186,159,186,122);arrow(320,159,320,134);arrow(320,134,186,134)
    arrow(239,99,268,99);arrow(373,99,400,99)
    arrow(53,122,53,139);arrow(53,139,453,139);arrow(453,139,453,122)
    box(134, 0, 239, 46, '结构化外力 f(X,t)', '36 个共享系数；进入原 PIDG 动量残差',True)
    box(0, 0, 106, 46, '原动量残差', 'r_legacy - f')
    arrow(134,23,106,23);arrow(53,77,53,46)
    return hh

def equation(text, name, height=38):
    fig=plt.figure(figsize=(9, .75));fig.patch.set_alpha(0)
    fig.text(.5,.5,text,ha='center',va='center',fontsize=17,color='#153149')
    path=TMP/(name+'.png');fig.savefig(path,dpi=220,transparent=True,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    return path

# Stable, GT-only union ROI for all selected times and all five models.
frame_ids = [0,25,49]
bounds=[]
for idx in frame_ids:
    arr=np.array(Image.open(RENDER/'A1/test/ours_best/gt'/f'{idx:05d}.png').convert('RGB'))
    yy,xx=np.where(arr.max(axis=2)>12)
    bounds.append((int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)))
roi=(max(0,min(b[0] for b in bounds)-30),max(0,min(b[1] for b in bounds)-30),
     min(1600,max(b[2] for b in bounds)+30),min(900,max(b[3] for b in bounds)+30))
pilfont=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',34)
pilbold=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',45)

def visual(idx):
    pw=960; ph=round(pw*(roi[3]-roi[1])/(roi[2]-roi[0])); gap=24; label=78
    sheet=Image.new('RGB',(pw*3+gap*2,(ph+label)*2+gap),'#f0f5f8')
    draw=ImageDraw.Draw(sheet)
    names=['GT']+models
    for k,name in enumerate(names):
        col=k%3;row=k//3;x=col*(pw+gap);y=row*(ph+label+gap)
        path=RENDER/('A1' if name=='GT' else name)/'test/ours_best'/('gt' if name=='GT' else 'renders')/f'{idx:05d}.png'
        im=Image.open(path).convert('RGB').crop(roi).resize((pw,ph),Image.Resampling.LANCZOS)
        draw.text((x+12,y+14),name,fill='#153149',font=pilbold)
        if name!='GT':
            draw.text((x+pw-290,y+22),f"best @ {best[name]:,}",fill='#637487',font=pilfont)
        sheet.paste(im,(x,y+label))
    path=TMP/f'comparison_{idx:05d}.png';sheet.save(path)
    return path

visuals={idx:visual(idx) for idx in frame_ids}

# Quantitative chart, using saved aggregate official-compatible image metrics.
all_names=['A0']+models
all_metrics=[baseline]+[r['metrics'] for r in rows]
chartcolors=['#8291A2','#70A9B3','#A6B5C1','#087F8C','#2D6D91','#BE8757']
fig,axes=plt.subplots(1,3,figsize=(11,4))
for ax,key,low,high in zip(axes,['PSNR','SSIM','LPIPS'],[25,.9745,.027],[29,.9777,.0375]):
    vals=[m[key] for m in all_metrics]
    ax.bar(np.arange(6),vals,color=chartcolors,width=.68)
    ax.axhline(vals[0],color='#637487',lw=.8,ls='--')
    ax.set_ylim(low,high);ax.set_xticks(np.arange(6),['A0','A1','A2a','A2b','A4_s','A6_s'],rotation=35,ha='right',fontsize=9)
    ax.set_title(key+(' (lower is better)' if key=='LPIPS' else ' (higher is better)'),fontsize=11,pad=12)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.tick_params(axis='y',labelsize=9)
fig.tight_layout(w_pad=2)
chart=TMP/'metrics.png';fig.savefig(chart,dpi=210,bbox_inches='tight');plt.close(fig)

# 1. Executive summary.
p=Page(1,'研究报告 / 执行摘要','PIDG 本构扩展实验报告','五个变体的结构、重建结果与物理诊断 | 数据冻结：2026 年 10 月 19 日')
p.text('基于 Rubber Duck 的 50k 长训练、原始兼容基线，以及已生成的 GT / Render 对照。本文按已保存结果撰写，不追加训练。',gap=18)
p.callout('<b>主要结果：</b>A2b（可学习共享材料参数）取得最高 PSNR：<b>28.409 dB</b>，相对复用基线提高 <b>1.152 dB</b>。基线的 LPIPS 仍最低；更复杂的外力与分阶段方案没有带来一致提升。')
p.heading('结构上的变化')
p.text('在保留 PIDG 高斯表示、形变场、速度 / 应力场和渲染器的基础上，新增独立的“局部形变梯度 F → Neo-Hookean 应力”分支，用一致性损失连接原应力预测。后续依次开放共享材料参数、低维外力及分阶段优化。')
p.table([['对象','本轮覆盖'],['长训练','A1、A2a、A2b、A4_single、A6_single，各 50,000 步'],['对照','复用 A0 原始兼容复现基线；不重复训练'],['渲染','同一组 50 个测试视图；GT 与五个模型逐帧对应'],['材料范围','真实场景使用单一共享材料；已知双材料仅做合成验证']], [88,CW-88])
p.heading('结论的边界')
p.text('结果支持：扩展模块已可运行，并在单次图像重建比较中观察到差异。结果尚不能证明材料参数或外力被准确恢复，也不能证明跨场景、跨随机种子的稳定收益。A1 不加入新损失，其相对旧基线的提升尤其不能直接归因于 F 模块。')
p.text('阅读索引：协议 02；结构 03-05；定量 06；GT 对照 07-09；物理诊断 10；可辨识性 11；合成验证 12；结论与来源 13。','small')
p.finish()

# 2. Protocol.
p=Page(2,'实验协议','实验设置与比较口径','本节定义训练预算、评估数据与快照选择，决定所有数值的解释方式。')
p.table([['项目','设置'],['场景 / 数据划分','Rubber Duck；150 个训练视图，50 个测试视图'],['训练预算','每个新变体 50,000 步；图像宽度 1,600；seed = 0；沿用 use_flow=False'],['重建评价','沿用项目 PSNR、SSIM、VGG-LPIPS、MS-SSIM 评估脚本'],['图像口径','表中为保存的 best 快照测试集均值，不是全部最终 50k 快照'],['物理口径','物理日志及条件 Jacobian 来自最终 50k 完整检查点'],['基线来源','复用已完成的 A0；历史运行未记录完整源码哈希'],['场景限制','未在本轮追加 D-NeRF / HyperNeRF；不将 NH 固体损失用于烟雾或布料']], [92,CW-92])
p.heading('最佳快照并不都在 50k')
p.table([['变体','训练完成步数','用于图像评估与渲染的快照']]+[[n,'50,000',f"{best[n]:,}"] for n in models],[112,137,CW-249])
p.text('历史协议使用测试 PSNR 选择 best。测试数据因而参与快照选择，本报告不将这些数值表述为独立保留集的无偏泛化估计。所有模型共用评估视图，仍便于进行本次协议内的描述性比较。','small')
p.text('A0 为原始发布代码的本地兼容复现结果。Windows / 发布代码兼容修复、历史源码记录不足以及 CUDA 非确定性，使它与新变体之间的差异不能全部归因于新增物理约束。来源：[1][2][3]。','small')
p.finish()

# 3. Architecture.
p=Page(3,'方法 / 总体结构','保留 PIDG，增加独立物理分支','所有新模块由配置开关控制；sigma_pred 保留，不以本构应力直接替换。')
p.check(242); c.saveState();c.translate(0,p.y-220);diagram();c.restoreState();p.y-=239
p.text('图 1  灰蓝色为沿用路径，浅绿色为新增模块。图中省略原有正则项；原 PIDG 速度、应力与动量约束继续保留。','caption')
p.text('沿用的骨干：Grid4D 分别编码 xyz 与 xyt / yzt / xzt 哈希特征，经空间门控和 MLP 输出中心变换、旋转及尺度偏移。速度 / 应力网络采用六平面特征、时间 Fourier 编码与共享 MLP，分别输出 3 维速度和 6 维对称应力。','small')
p.table([['变体','F / 本构','材料参数','外力','优化方式'],['A1','仅 F 诊断','无新增学习','关','原训练'],['A2a','F + NH 一致性','固定 μ=1, κ=10','关','联合'],['A2b','F + NH 一致性','共享 μ、κ 可学习','关','联合'],['A4_single','同 A2b','单一共享材料','低秩，共 36 参数','联合'],['A6_single','同 A4_single','单一共享材料','同 A4_single','A-B-C-D 分阶段']], [72,119,120,99,CW-410])
p.text('A1 的 F 只用于记录诊断，不加入新损失，也不建立可学习材料 / 外力变量。A2b 相比 A2a 仅开放两个共享材料标量；A4_single 再引入 36 个场景共享外力系数。','small')
p.text('A3 的已知材料区域与界面模块尚无 Rubber Duck 真实标签，未列为真实场景长训练。A5 是训练后的可辨识性分析，复用 A4 的模型，故本次可视化共有五个新模型。来源：[3][4]。','small')
p.finish()

# 4. Physics branch equations.
p=Page(4,'方法 / 形变与本构','从局部形变到应力一致性','F 描述局部材料坐标如何变形；本构分支将几何状态连接到共享材料参数。')
p.heading('1. 可微分的局部形变梯度')
p.text('从规范高斯中心 X 与渲染时实际中心 x(t) 构造邻域差分。实际中心由形变网络的四元数旋转和平移给出：x(t)=R(q)X+b。使用 16 个规范空间近邻和距离权重做带单位先验的局部最小二乘。')
p.image(equation(r'$A=\sum_j w_j\Delta X_j\Delta X_j^T,\quad B=\sum_j w_j\Delta x_j\Delta X_j^T$', 'eq_F1'),38)
p.image(equation(r'$F=(B+\alpha I)(A+\alpha I)^{-1},\quad J=\det(F)$', 'eq_F2'),38)
p.text('α 为局部尺度相关的微小岭项。少于 6 个有效邻居、未正则化条件数过大、非有限值或 J≤10^-6 的查询被屏蔽；置信度不参与反向传播。规范邻接图按周期或点数变化更新。')
p.heading('2. 可压缩 Neo-Hookean 固体应力')
p.image(equation(r'$\sigma_{const}=\mu J^{-5/3}\operatorname{dev}(FF^T)+\kappa\frac{\ln J}{J}I$', 'eq_NH'),43)
p.text('μ 是剪切参数，κ 是体积参数。可学习版本通过 softplus(raw)+10^-6 保持为正；本轮对整个场景共享一组参数，初值 (1,10)，没有逐高斯自由刚度。刚体旋转对应近零弹性应力，已由模块测试覆盖。')
p.heading('3. 稀疏计算的一致性损失')
p.text('保留的 σ_pred 包含原对称应力输出及速度梯度修正。对 σ_pred 与 σ_const 使用逐矩阵元素 Huber 距离，并乘有效掩码及固定置信度；随后对全部查询求均值。新增权重为 10^-4，预热后每 100 步对至多 256 个查询计算一次。')
p.callout('该损失可向 F、原应力分支和开放的材料参数传梯度。无效样本贡献为零；它们减少有效约束量，不能把“损失变小”直接解释为更准确的材料恢复。来源：[4]。')
p.finish()

# 5. Forces and schedule.
p=Page(5,'方法 / 外力与优化','结构化外力与分阶段训练','A4_single 在 A2b 上加入外力；A6_single 保留结构并修改参数更新日程。')
p.image(equation(r'$f(X,t)=\sum_{r=1}^{3}\phi_r(X)\,a_r(t),\qquad \phi=[1,X_x,X_y]$', 'eq_force'),40)
p.text('a_r(t) 是三维向量，在归一化时间的 4 个控制点间线性插值。因此共有 4×3×3=36 个共享系数，初始化为零。它约束外力的空间与时间结构，不引入逐高斯自由力。其空间基仅含常数和 X、Y 一次项。')
p.text('外力作为体积力密度进入 r_legacy - f。另加幅度正则 10^-4 和时间系数差分正则 10^-3；本次稀疏项权重为 0。A4 的材料学习率为 10^-5，外力为 10^-4。')
p.heading('A6 的实际更新日程')
p.table([['阶段 / 迭代','原模型参数','材料 / 外力','新增约束'],['A / 1-4,999','原学习率更新','冻结 / 关闭外力','不加新损失'],['B / 5,000-9,999','原学习率更新','材料固定 / 外力关闭','本构一致性'],['C / 10,000-19,999','全部冻结；停用增密','μ、κ 与外力更新','本构 + 外力残差'],['D / 20,000-50,000','学习率乘 0.1','材料及外力学习率再乘 0.1','联合细化']], [115,130,138,CW-383])
p.text('阶段 C 的神经参数进入 / 退出哈希一致，日志记录为 true。阶段 D 开始时已超过原增密截止 15k，不能补回阶段 C 期间被跳过的增密机会。50k 相同迭代数并不意味着神经模型获得相同更新预算。','small')
p.callout('<b>物理解释限制：</b>为保留发布版路径，原速度 / 动量残差继续在 fid/train_count 查询；渲染 F、本构应力与新外力使用 fid。二者时间口径不同，本报告不将该残差等同于已校准真实物理单位下的受力平衡。来源：[4][5]。')
p.finish()

# 6. Quantitative metrics.
p=Page(6,'结果 / 定量重建','A2b 的 PSNR 最高，收益存在取舍','↑ 越高越好；↓ 越低越好。全部数值为同一测试协议下保存的 best 快照均值。')
data=[['方法','PSNR ↑','ΔPSNR','SSIM ↑','LPIPS ↓','MS-SSIM ↑']]
for name,m in zip(all_names,all_metrics):
    data.append([name,f"{m['PSNR']:.3f}",f"{m['PSNR']-baseline['PSNR']:+.3f}",f"{m['SSIM']:.6f}",f"{m['LPIPS']:.6f}",f"{m['MS-SSIM']:.6f}"])
p.table(data,[90,69,66,92,93,CW-410],highlight=4)
p.image(chart,190,'图 2  三项重建指标。虚线为 A0；纵轴截断用于显示小差异。A4_s、A6_s 对应 single 版本。无重复实验，故不绘制误差条。')
p.text('<b>可学习材料对照：</b>A2b 比固定材料 A2a 高 1.281 dB；相对 A0 高 1.152 dB。但 A2b 的 LPIPS 为 0.030052，仍高于 A0 的 0.029624，不能宣称所有图像指标全面提升。')
p.text('<b>增加复杂度的效果：</b>A4_single 比 A2b 低 0.080 dB；A6_single 比 A4_single 低 2.002 dB。分阶段版本在本次重建中较弱，原因不能仅凭这一次实验归结为某个物理模块。')
p.text('<b>A1 的重要对照：</b>A1 不增加训练损失，却比旧基线高 0.893 dB。这表明需补做同版本 A0、多随机种子及统一快照选择，再判断新增约束的因果收益。来源：[1][2]。','small')
p.finish()

# 7-9. Visual comparisons: start / middle / end, fixed GT-only ROI.
comments={0:('起始时刻','初始时刻可检查鸭体与蓝色支架的空间布局。各变体保留了主体结构，但轮廓、支架细边及小部件与 GT 仍有差异；单帧不能替代全测试集平均结果。'),
          25:('中间时刻','中间时刻用于检查变形过程中鸭体轮廓和支架边缘。模型整体布局接近 GT，细节锐度及局部颜色仍有偏差。应结合连续视频判断抖动或时间一致性，静态图不足以量化这些问题。'),
          49:('末尾时刻','末尾时刻展示归一化时间 1.0 的输出。各模型仍可重建主要物体，但局部细节误差持续存在。这里属于测试快照协议内的重建展示，不构成独立未来预测或真实动力学恢复证据。')}
for number,idx in enumerate(frame_ids,7):
    stage,comment=comments[idx]
    view=manifest['views'][idx]
    p=Page(number,'结果 / GT 与五模型',f'GT 对照：{stage}',f"测试索引 {idx:02d} / 49 | 原图 {view['image_name']} | 归一化时间 t={view['time']:.3f}")
    p.text('预先取起始、中间、末尾三个视图。每个面板使用同一个由三张 GT 前景并集确定的裁剪窗口；所有模型采用相同缩放，未进行颜色增强。','small')
    im=Image.open(visuals[idx]);height=min(462,CW*im.height/im.width)
    p.image(visuals[idx],height,f'图 {number-4}  GT 与五个变体的固定窗口对照。面板标注为各自 best 快照步数；原始全分辨率视图为 1600×900。')
    p.text(comment)
    p.text('当前五模型图集未包含 A0 的同面板重渲染，故基线只在定量表中比较。完整图集含 50 帧、每模型 GT/Render 拼接 PNG 与 10 fps 视频；面板与旧评估渲染逐像素一致。来源：[6]。','small')
    p.finish()

# 10. Physical diagnostics.
p=Page(10,'结果 / 最终检查点','图像提升尚未对应材料参数恢复','本页仅使用最终 50k 日志；与第 6 页 best 快照指标属于不同快照口径。')
data=[['变体','μ','κ','原始本构损失','无效 F 比例']]
for n in models:
    ph=by_name[n]['physics']
    data.append([n,f"{ph['mu']:.6f}" if 'mu'in ph else '不适用',f"{ph['kappa']:.6f}" if 'kappa'in ph else '不适用',f"{ph['loss_constitutive_raw']:.6f}" if 'loss_constitutive_raw'in ph else '未加入',f"{100*ph['invalid_F_fraction']:.4f}%"])
p.table(data,[100,88,95,125,CW-408])
p.text('可学习材料的最终数值仍非常接近初值 μ=1、κ=10。例如 A2b 的 μ 仅相对初值变化约 0.050%，κ 约 0.017%。在没有真实材料标签的 Rubber Duck 上，无法由这些数字检验真实参数误差。')
p.heading('外力与应力的日志量级')
data=[['最终 50k 日志','A4_single','A6_single']]
for label,key in [('平均本构应力范数','mean_sigma_const_norm'),('平均预测应力范数','mean_sigma_pred_norm'),('平均应力差范数','mean_sigma_difference_norm'),('外力项范数','external_force_norm'),('残差项范数','total_residual_norm')]:
    data.append([label]+[f"{by_name[n]['physics'][key]:.6f}" for n in ['A4_single','A6_single']])
p.table(data,[239,134,CW-373])
p.text('A6 的本构损失和记录残差较小，但图像重建指标较差，说明优化这些代理量不保证重建质量单调改善。应力分支也未被一致性项完全对齐；这些日志不能直接比较真实力学精度。')
p.text('日志记录了本构损失到 F 与预测应力的非零有限梯度，支持梯度路径已连接。例如 A2b 在 50k 的两项梯度范数分别约 3.54×10^-5 与 8.86×10^-7。有效性统计每次只抽样 256 个点；0% 无效不能证明全场均无反转。来源：[1][4]。','small')
p.finish()

# 11. Identifiability.
p=Page(11,'结果 / 可辨识性','条件满秩不等于从图像识别出真实物理','诊断冻结神经几何、速度和应力，只考察少量物理参数的局部残差 Jacobian。')
diag4=by_name['A4_single']['identifiability']['conditional_physical_residual']
diag6=by_name['A6_single']['identifiability']['conditional_physical_residual']
p.table([['最终检查点','条件秩 / 参数数','条件数','最小奇异值'],['A4_single','38 / 38',f"{diag4['condition_number']:,.1f}",f"{diag4['min_singular_value']:.4f}"],['A6_single','38 / 38',f"{diag6['condition_number']:,.1f}",f"{diag6['min_singular_value']:.4f}"]],[118,145,126,CW-389])
p.text('38 个参数为 log μ、log κ 和 36 个外力系数。结果依赖固定的神经场、残差单位、参数化和数值尺度；未把这些神经场的不确定性边缘化，也不是经过噪声校准的置信区间。')
p.callout('特别地，当前诊断将“应力一致性残差”与“原动量残差减外力”拼接。在冻结神经场后，材料参数只影响前一块，外力只影响后一块，因此材料 / 外力交叉相关为 0 是该条件构造的结构性结果，不能解释为真实未知量之间不存在混淆。')
p.heading('独立的合成歧义实验提供反例')
p.text('制造平衡 μu+κv-fu=y，其中 u=sin(t)、v=cos(t)，真值 (μ,κ,f)=(2,3,1)。当 μ 与 f 同时未知，观测只能约束其差值。该实验为解析平衡测试，并非从官方 RGB 图像反演。')
p.table([['设置','数据秩 / 参数数','目标秩','μ 相对误差','力相对误差'],['已知外力','2 / 2','2','0%','0%'],['已知刚度','1 / 1','1','0%','0%'],['二者未知','2 / 3','2','75%','150%'],['二者未知 + 偏置先验','2 / 3','3','25%','50%']],[136,110,62,96,CW-404])
p.text('二者未知时条件数为无穷大，材料 / 外力相关代理约 0.999978。加入偏置先验后，目标函数条件数改善至 20.55，但观测数据仍秩亏且参数仍有偏差。优化收敛、正则化后的满秩和物理恢复是不同结论。来源：[5][7]。','small')
p.finish()

# 12. Controlled validation.
p=Page(12,'验证 / 合成双材料','多材料与界面：已验证模块，尚待真实标签','本页是工程验证。300 步、宽度 400 的制造数据不能并入 50k 官方场景排名。')
p.text('制造数据包含两个明确给定材料区域，共 128 个材料点；多材料版本采用共享材料库，不学习分割。真值 μ=[1,2]、κ=[10,20]，多材料配置从低于真值 20% 的初值开始；图像由简化圆盘投影生成。界面法向使用规范中心连线代理。')
data=[['合成配置','PSNR ↑','SSIM ↑','LPIPS ↓','界面代理损失']]
for row in results['smoke_rows']:
    n=row['configuration_id'].replace('_controlled','')
    ph=row.get('physics',{})
    interface=ph.get('loss_interface_raw',ph.get('interface_traction_raw'))
    # Saved rows use the actual per-run field below; no absent metric is zero-filled.
    if interface is None:
        candidates=[v for k,v in ph.items() if 'interface' in k and isinstance(v,(int,float))]
        interface=candidates[0] if candidates else None
    data.append([n,f"{row['metrics']['PSNR']:.3f}",f"{row['metrics']['SSIM']:.6f}",f"{row['metrics']['LPIPS']:.6f}",f"{interface:.6f}" if interface is not None else '未记录 / 未启用'])
p.table(data,[145,72,91,91,CW-399])
p.text('图像指标没有随模块数量单调提高。该组对照验证不同配置能够独立运行及产生日志；不能据此确认材料或外力已恢复。真实多材料的 A3，以及带真实界面标签的完整 A4/A6，仍需等待 material_id。')
p.heading('测试与恢复证据的区别')
p.text('保存的里程碑验证覆盖：仿射场 F 恢复、刚体旋转近零 NH 应力、本构反向传播、不同区域共享参数、外力有限梯度、阶段冻结和开关关闭路径等。M9 早期独立合成 smoke 中，条件 Jacobian 为 40/40 满秩，材料误差却仍约 20%，外力相对误差约 100%。')
p.text('这说明“模块测试通过”与“真实物理参数识别成功”应分开报告。真实 Rubber Duck 没有可用材料 / 外力真值，本报告不填入其物理恢复误差。来源：[1][3][7][8]。','small')
p.finish()

# 13. Conclusions and local sources.
p=Page(13,'结论 / 来源','当前结论与下一步验证','保留有希望的结果，同时用公平对照分离模型结构、优化预算和运行差异。')
p.text('<b>模型选择：</b>若本轮以 PSNR 优先，A2b 是最有希望的候选；若以 LPIPS 优先，复用 A0 仍最好。A4_single 的额外外力未提高 PSNR；A6_single 当前配置不宜作为重建性能的首选。')
p.text('<b>解释：</b>本构分支可以作为稀疏的几何 / 应力一致性约束，但材料几乎停留在初值、不同时间口径及条件诊断限制，都不支持“真实材料与外力已恢复”的结论。A1 的差异也提示需先排除非确定性与历史基线差异。')
p.text('<b>建议顺序：</b>① 同版本、同环境重跑 flags-off A0 与 A1，采用独立验证集选快照并重复多种子；② 在此基础上复核 A2a/A2b 的收益及损失频率 / 权重；③ 比较 A4 与 A6 时匹配神经更新和增密预算；④ 另立实验统一物理时间与量纲，再用已知真值 / 标签检验材料、外力和联合可辨识性。')
p.heading('本地来源与复现入口')
sources=[
    '[1] experiments/results.json、results.md：长训练、合成对照、最终物理日志摘要。',
    '[2] baseline_metrics.json；PIDG_REPO_MAP.md：兼容基线数值、代码映射与已知差异。',
    '[3] experiments/configs/*.json；pidg_experiment_matrix.yaml：变体开关与场景范围。',
    '[4] pidg_ext/deformation_gradient.py、constitutive.py、losses.py、runtime.py：F、NH、损失与训练集成。',
    '[5] pidg_ext/forces.py、staging.py、identifiability.py；scripts/diagnose_constitutive.py：外力、日程及条件诊断定义。',
    '[6] runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/manifest.json 与 delivery_validation.json：快照、视图与像素校验。',
    '[7] experiments/M9_validation.json；runs/pidg_constitutive/m10_long_A4_single/rubber_duck/identifiability.json 与 A6 对应文件：歧义与最终诊断。',
    '[8] experiments/M1_validation.json 至 M10_validation.json 中已有记录；experiments/milestones.json：里程碑验证和状态。',
]
for line in sources:p.text(escape(line),'small',gap=6)
p.text('报告生成脚本：scripts/build_pidg_report.py。图表直接读取保存的 JSON；图像来自已校验 PNG。完整 GT/Render 图集保留在 [6] 同目录的 index.html、各模型 gt_render.mp4 及 gt_and_five_models.mp4。','small',gap=0)
p.finish()
c.save()

provenance_paths=[ROOT/'experiments/results.json',ROOT/'baseline_metrics.json',ROOT/'PIDG_REPO_MAP.md',
                  ROOT/'scripts/diagnose_constitutive.py',ROOT/'experiments/milestones.json',
                  ROOT/'runs/pidg_constitutive/m10_long_A6_single/rubber_duck/optimization_stages.jsonl',
                  RENDER/'manifest.json',RENDER/'delivery_validation.json',Path(__file__)]
provenance_paths+=sorted((ROOT/'pidg_ext').glob('*.py'))+sorted((ROOT/'experiments/configs').glob('*.json'))
provenance_paths+=sorted((ROOT/'experiments').glob('M*_validation.json'))
provenance_paths += [ROOT/'variants/constitutive_pidg/code/scene_PIDG/network.py',
                    ROOT/'variants/constitutive_pidg/code/motion_utils/pidg.py']
provenance={'pdf':str(PDF),'pages':page_records,'fixed_gt_roi_xyxy':roi,'visual_indices':frame_ids,
            'best_snapshot_iterations':best,'source_sha256':{str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in provenance_paths},
            'limitations':['one scene, one seed','best selected by test PSNR','A0 historical source hashes unavailable',
                           'image best snapshot differs from final physics checkpoint','conditional Jacobian freezes neural fields',
                           'legacy CMR and new constitutive/force time conventions differ']}
(OUT/'PIDG_Variants_Report.provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(PDF),'pages':len(page_records),'roi':roi,'content_bottom':[q['content_bottom'] for q in page_records]},ensure_ascii=False))
