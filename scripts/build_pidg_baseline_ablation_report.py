"""Build a Chinese report of completed baseline MLP and missing-frame experiments.
Reads saved results and images only; does not import Torch or modify any run.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
TMP = ROOT / 'tmp/pdfs/pidg_baseline_report'
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
PDF = OUT / 'PIDG_MLP_MaskedData_Report.pdf'
REPORT_TIME = datetime.now(timezone.utc).astimezone(ZoneInfo('Europe/London'))

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

mlp = read(ROOT / 'experiments/mlp_capacity/results.json')
mask = read(ROOT / 'experiments/random_frame_mask/results.json')
mask_plan = read(ROOT / 'experiments/random_frame_mask/plan.json')
mlp_rows, mask_rows = mlp['rows'], mask['rows']
assert mlp['controller']['status'] == mask['controller']['status'] == 'completed'
assert all(r['status'].startswith('completed') for r in mlp_rows + mask_rows)
base = mlp_rows[0]['metrics']
original_test = read(ROOT / 'data/PIDG/rubber_duck/transforms_test.json')
render_catalog, run_records = [], []

def inspect_run(row, family):
    run = Path(row['run'])
    info = read(run / 'reproduction.json')
    assert info['status'] == 'completed' and info['iterations'] == 50000
    assert info['resolution_width'] == 1600 and info['seed'] == 0
    metric = read(run / 'results.json')['ours_best']
    saved = row['metrics'] if family == 'mlp' else row['official_metrics']
    assert metric == saved
    step = int(re.findall(r'\d+', (run / 'deform/iteration_best/iter.txt').read_text())[-1])
    key = row.get('id', str(row.get('drop_percent')) + '%')
    def add(kind, directory, count):
        directory = directory.absolute()
        gt = directory.parent / 'gt'
        expected = {f'{i:05d}.png' for i in range(count)}
        assert {p.name for p in directory.glob('*.png')} == expected
        assert {p.name for p in gt.glob('*.png')} == expected
        item = {'family':family,'label':key,'split':kind,'render_absolute':str(directory),
                'gt_absolute':str(gt),'frame_count':count,'best_iteration':step}
        render_catalog.append(item)
    add('official_test', run / 'test/ours_best/renders', 50)
    if family == 'mask' and row['drop_percent'] > 0:
        assert {'train','render','metrics','heldout_render','heldout_metrics'}.issubset(info['completed_steps'])
        assert read(run / 'masked_holdout/results.json')['ours_best'] == row['masked_holdout_metrics']
        add('dropped_training_frames', run / 'masked_holdout/test/ours_best/renders', row['removed'])
    run_records.append({'family':family,'label':key,'run':str(run),'best_iteration':step,
                        'reproduction_sha256':digest(run / 'reproduction.json')})
    return step

mlp_best = {r['id']:inspect_run(r,'mlp') for r in mlp_rows}
mask_best = {r['drop_percent']:inspect_run(r,'mask') for r in mask_rows if r['drop_percent']>0}
split_path = ROOT / 'experiments/random_frame_mask/split_metrics.json'
split_results = read(split_path) if split_path.exists() else None
if split_results is not None:
    assert split_results['status'] == 'completed', 'Split evaluation must finish before rebuilding the report'
    split_validation = read(ROOT / 'experiments/random_frame_mask/split_metrics_validation.json')
    assert split_validation['status'] == 'passed' and split_validation['evaluated_view_model_pairs'] == 1200
    assert [r['drop_percent'] for r in split_results['rows']] == [0, 10, 30, 50, 70, 90]
    for row in split_results['rows']:
        count = row['splits']['retained']['count']
        directory = Path(row['retained_render_absolute'])
        gt = Path(row['retained_gt_absolute'])
        expected = {f'{i:05d}.png' for i in range(count)}
        assert {p.name for p in directory.glob('*.png')} == expected
        assert {p.name for p in gt.glob('*.png')} == expected
        assert row['splits']['original_train']['count'] == 150 and row['splits']['all_200']['count'] == 200
        render_catalog.append({'family':'retained_eval','label':str(row['drop_percent'])+'%',
                               'split':'retained_training_frames','render_absolute':str(directory),
                               'gt_absolute':str(gt),'frame_count':count,'best_iteration':row['best_iteration']})
TOTAL_PAGES = 16 if split_results else 12

pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('CN-Bold','C:/Windows/Fonts/msyhbd.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('CN',normal='CN',bold='CN-Bold',italic='CN',boldItalic='CN-Bold')
pdfmetrics.registerFont(TTFont('Arial','C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('PathFont','C:/Windows/Fonts/consola.ttf'))
NAVY=colors.HexColor('#153149');TEAL=colors.HexColor('#087F8C');INK=colors.HexColor('#253649')
GRAY=colors.HexColor('#637487');PALE=colors.HexColor('#EFF5F8')
W,H=595.276,841.89
LEFT=RIGHT=44
CW=W-LEFT-RIGHT
styles={}
for name,size,leading,color in [('body',10.2,17,INK),('small',8.5,13,GRAY),
                              ('caption',8.5,13,GRAY),('table',8.3,12.5,INK),('thead',8.3,12.5,colors.white),
                              ('path',7.8,11,INK)]:
    styles[name]=ParagraphStyle(name,fontName='PathFont' if name=='path' else 'CN',fontSize=size,
                               leading=leading,textColor=color,wordWrap='CJK')
c=canvas.Canvas(str(PDF),pagesize=(W,H),pageCompression=1)
c.setTitle('PIDG baseline：MLP 容量与随机缺帧实验报告')
c.setAuthor('PhysGauss / PIDG Extension Project')
c.setSubject('Completed experiments, GT/render comparisons and absolute render directories')
page_records=[]
def P(text,style='body'):
    return Paragraph(text,styles[style])

def path_markup(value):
    """Wrap long Windows paths at directory boundaries, preserving every character."""
    lines, current = [], ''
    parts = value.split('\\')
    for index, part in enumerate(parts):
        token = part + ('\\' if index < len(parts) - 1 else '')
        if current and pdfmetrics.stringWidth(current + token, 'PathFont', 7.8) > CW:
            lines.append(current)
            current = ''
        current += token
    if current:
        lines.append(current)
    return '<br/>'.join(escape(line) for line in lines)

class Page:
    def __init__(self, number, label, title, subtitle=''):
        self.number, self.label, self.title = number, label, title
        c.setFillColor(NAVY)
        c.rect(0, H - 11, W, 11, fill=1, stroke=0)
        c.setFont('Arial', 8)
        c.setFillColor(TEAL)
        c.drawString(LEFT, H - 39, 'PIDG / BASELINE ABLATIONS')
        c.setFont('CN', 8)
        c.setFillColor(GRAY)
        c.drawRightString(W - RIGHT, H - 39, label)
        title_size = min(22, 22 * CW / max(CW, pdfmetrics.stringWidth(title, 'CN-Bold', 22)))
        c.setFont('CN-Bold', title_size)
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
        c.drawString(LEFT, 30, 'Rubber Duck | 50k | 单场景、单随机种子 | '+REPORT_TIME.strftime('%Y-%m-%d'))
        c.setFont('Arial', 8)
        c.drawRightString(W - RIGHT, 30, f'{self.number:02d} / {TOTAL_PAGES}')
        page_records.append({'page': self.number, 'title': self.title, 'content_bottom': round(self.y, 2)})
        c.showPage()


def chart_file(fig, name):
    path = TMP / (name + '.png')
    fig.savefig(path, dpi=210, bbox_inches='tight'); plt.close(fig)
    return path


fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6))
counts = [r['mlp_parameters'] for r in mlp_rows]
scores = [r['metrics']['PSNR'] for r in mlp_rows]
axes[0].plot(counts, scores, color='#087F8C', marker='o', lw=1.3)
axes[0].set_xscale('log'); axes[0].set_ylim(26.9, 29)
axes[0].set_xlabel('Deformation MLP parameters (log scale)'); axes[0].set_ylabel('PSNR (dB)')
for i, r in enumerate(mlp_rows):
    axes[0].annotate(r['id'], (counts[i], scores[i]), xytext=(-80, 8) if i==3 else (6, 5),
                     textcoords='offset points', fontsize=8.5)
axes[1].bar(['256 / 0','512 / 0','256 / 2','512 / 2'], [r['metrics']['LPIPS'] for r in mlp_rows],
            color=['#8395A5','#087F8C','#2D6D91','#BA8756'])
axes[1].set_ylim(.027,.037);axes[1].set_ylabel('LPIPS (lower is better)');axes[1].set_xlabel('Width / depth')
for ax in axes:
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.tight_layout(w_pad=3)
mlp_chart = chart_file(fig, 'mlp_metrics')

fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6))
percent = [r['drop_percent'] for r in mask_rows]
for ax, key, label in zip(axes, ['PSNR','LPIPS'], ['PSNR (dB)','LPIPS (lower is better)']):
    vals=[r['official_metrics'][key] for r in mask_rows]
    ax.plot(percent,vals,marker='o',color='#087F8C',lw=1.6)
    ax.axhline(vals[0],color='#8395A5',ls='--',lw=.8)
    ax.set_xticks(percent);ax.set_xlabel('Training frames removed (%)');ax.set_ylabel(label)
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
fig.tight_layout(w_pad=3)
mask_chart = chart_file(fig, 'mask_official_metrics')

fig, ax = plt.subplots(figsize=(10.8, 3.2))
ax.plot(percent[1:],[r['official_metrics']['PSNR'] for r in mask_rows[1:]],marker='o',color='#087F8C',label='Fixed official test: 50 views')
ax.plot(percent[1:],[r['masked_holdout_metrics']['PSNR'] for r in mask_rows[1:]],marker='s',color='#B67B45',label='Dropped-frame holdout: membership varies')
ax.set_xticks(percent[1:]);ax.set_xlabel('Training frames removed (%)');ax.set_ylabel('PSNR (dB)')
ax.legend(frameon=False,fontsize=9);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
fig.tight_layout()
heldout_chart = chart_file(fig, 'mask_holdout_metrics')

image_checks=[]
font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',45)
subfont=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',33)


def comparison(entries, name, columns=3):
    # Entries contain a render and its exact matching GT at the same camera/time.
    gt=Image.open(entries[0]['gt']).convert('RGB')
    assert gt.size==(1600,900)
    for entry in entries:
        current=Image.open(entry['gt']).convert('RGB')
        assert current.size==gt.size and current.tobytes()==gt.tobytes(), entry['label']
    arr=np.array(gt);yy,xx=np.where(arr.max(axis=2)>12)
    roi=(max(0,int(xx.min())-24),max(0,int(yy.min())-24),
         min(1600,int(xx.max())+25),min(900,int(yy.max())+25))
    pw=960;ph=round(pw*(roi[3]-roi[1])/(roi[2]-roi[0]));gap=24;label=82
    panels=[{'label':'GT','render':entries[0]['gt'],'subtitle':'same camera and time'}]+entries
    nrows=(len(panels)+columns-1)//columns
    canvas_img=Image.new('RGB',(columns*pw+(columns-1)*gap,nrows*(ph+label)+(nrows-1)*gap),'#eff5f8')
    draw=ImageDraw.Draw(canvas_img)
    for i,entry in enumerate(panels):
        x=(i%columns)*(pw+gap);y=(i//columns)*(ph+label+gap)
        draw.text((x+12,y+14),entry['label'],font=font,fill='#153149')
        draw.text((x+pw-360,y+24),entry.get('subtitle',''),font=subfont,fill='#637487')
        image=Image.open(entry['render']).convert('RGB')
        assert image.size==gt.size
        image=image.crop(roi).resize((pw,ph),Image.Resampling.LANCZOS)
        canvas_img.paste(image,(x,y+label))
    path=TMP/(name+'.png');canvas_img.save(path)
    image_checks.append({'figure':name,'roi_xyxy':roi,'gt_pixel_identity':True,
                         'panels':[{'label':e['label'],'render':str(e['render']),'gt':str(e['gt']),
                                    'render_sha256':digest(e['render']),'gt_sha256':digest(e['gt'])} for e in entries]})
    return path


def official_entry(row, label, best):
    basepath=Path(row['run'])/'test/ours_best'
    return {'label':label,'render':basepath/'renders/00025.png','gt':basepath/'gt/00025.png',
            'subtitle':f'best @ {best:,}'}


mlp_visual=comparison([official_entry(r,r['id'],mlp_best[r['id']]) for r in mlp_rows], 'mlp_visual',3)
low_visual=comparison([official_entry(r,'Baseline 0%' if r['drop_percent']==0 else f"Drop {r['drop_percent']}%",
                                     50000 if r['drop_percent']==0 else mask_best[r['drop_percent']])
                       for r in mask_rows if r['drop_percent'] in (0,10,30)], 'mask_low_visual',2)
high_visual=comparison([official_entry(r,f"Drop {r['drop_percent']}%",mask_best[r['drop_percent']])
                        for r in mask_rows if r['drop_percent'] in (50,70,90)], 'mask_high_visual',2)

# Match one genuinely withheld frame across all ratios by source identity, never by PNG index.
first_doc=read(ROOT/'data/pidg_random_frame_mask/rubber_duck/drop10_seed0/heldout_dataset/transforms_test.json')
common_frame=first_doc['frames'][len(first_doc['frames'])//2]
common_id=common_frame['file_path']
heldout_entries=[];heldout_indices={}
for row in mask_rows[1:]:
    ratio=row['drop_percent']
    doc=read(ROOT/f'data/pidg_random_frame_mask/rubber_duck/drop{ratio:02d}_seed0/heldout_dataset/transforms_test.json')
    idx=next(i for i,f in enumerate(doc['frames']) if f['file_path']==common_id)
    assert doc['frames'][idx]==common_frame
    heldout_indices[ratio]=idx
    path=Path(row['run'])/'masked_holdout/test/ours_best'
    heldout_entries.append({'label':f'Drop {ratio}%','render':path/'renders'/f'{idx:05d}.png',
                            'gt':path/'gt'/f'{idx:05d}.png','subtitle':f'index {idx:02d}'})
heldout_visual=comparison(heldout_entries,'heldout_visual',3)


def draw_visual(page,path,caption,max_height=430):
    im=Image.open(path)
    page.image(path,min(max_height,CW*im.height/im.width),caption)


# 1. Overview.
p=Page(1,'完成结果 / 执行摘要','PIDG baseline 容量与缺帧实验报告',
       'MLP 参数量消融 + 随机训练帧移除 | '+REPORT_TIME.strftime('%Y-%m-%d %H:%M %Z'))
p.text('本报告汇总两组已完成实验：三个新增 MLP 容量变体，以及五档随机缺帧 baseline。所有新增长训练均为 Rubber Duck、50k、宽度 1600、训练 seed0；完整数据的原始 baseline 复用已有结果。')
p.callout('<b>MLP 容量：</b>512×0 取得最高 PSNR <b>28.577 dB</b>，相对原始 baseline 提高 <b>1.320 dB</b>。最大参数量的 512×2 没有最好表现；LPIPS 仍以原始 baseline 最低。')
p.callout('<b>随机缺帧：</b>10% 缺帧的官方测试 PSNR 为 <b>28.253 dB</b>；90% 缺帧降至 <b>23.063 dB</b>，比完整数据 baseline 低 <b>4.194 dB</b>。单次运行中间各档没有严格单调变化。')
p.table([['实验','完成情况'],['MLP 容量','256×0 原始对照复用；512×0、256×2、512×2 均训练、渲染、评估完成'],['随机缺帧','10%、30%、50%、70%、90% 均完成官方测试与被移除帧的渲染、评估'],['顺序依赖','MLP 于 10 月 20 日 18:14 BST 完成；缺帧队列随后自动启动，并于同日 22:49 BST 完成'],['图片核验',f"{len(render_catalog)} 个 render 目录，共 {sum(x['frame_count'] for x in render_catalog):,} 张 render 与匹配 GT，文件名与数量已核对"]], [87,CW-87])
p.text('两组均使用原始 PIDG 兼容 baseline，未启用新增 F、NH、材料参数学习或结构化外力分支。结果是单场景、单随机种子的描述性比较，尚不能证明统计显著提升。')
p.text('阅读索引：设置 02；MLP 数值与图像 03-04；缺帧数值 05-06；图像对照 07-09；render 绝对地址 10-11；结论与来源 12。','small')
if split_results:
    p.callout('<b>补充评估 13-16 页：</b>原主表只统计 50 张官方测试图。本版补算保留训练帧，并将保留 / 被移除帧分别及合并统计，另提供完整 200 帧指标与新增 render 绝对地址。')
p.finish()

# 2. Protocol.
p=Page(2,'方法 / 实验协议','两组实验的控制变量与评估口径','原始 baseline 网络与损失保留；新增物理分支全部关闭。')
p.text('形变骨干为 Grid4D 多分辨率哈希特征、空间门控与 MLP 输出头；原速度 / 应力网络和动量损失继续保留。容量实验只改变现有 DeformNetwork 的宽度 w 与额外隐藏层数 d；哈希网格、速度 / 应力网络、学习率规则和增密设置一致。')
p.table([['MLP 配置','宽度 / 深度','形变 MLP 参数量','相对原始 MLP'],
         *[[r['id'],f"{r['width']} / {r['depth']}",f"{r['mlp_parameters']:,}",f"{r['mlp_parameters']/61454:.2f} 倍"] for r in mlp_rows]],
        [111,110,151,CW-372])
p.heading('缺帧：固定 seed0 的嵌套随机子集')
p.table([['移除比例','保留训练帧','独立被移除帧','训练时间覆盖'],
         *[[f"{j['drop_percent']}%",j['retained_count'],j['dropped_count'],f"{j['retained_time_min']:.3f} 至 {j['retained_time_max']:.3f}"] for j in mask_plan['jobs']]],
        [83,115,139,CW-337])
p.text('从原始 150 张训练图像中随机移除整帧。较高比例包含较低比例已移除的帧；保留原顺序、时间戳、相机位姿及规范点云。原始 RGB、mask、光流文件不删除，仅由新清单决定是否进入训练。')
p.text('官方 50 张测试图像在所有配置中保持一致。沿用原 best-test-PSNR 快照选择；被移除帧只在训练结束后评估，不参与选择。50k 是训练完成步数，渲染快照可能更早。固定 50k 预算使高缺帧时每张保留图像被重复更多次。','small')
p.text('原始 PIDG 的动量查询使用 fid/train_count；保留训练帧减少时，该原生归一化也随之改变。本轮未引入新的物理修正，因此测量包括 baseline 对减少训练数据的整体响应。来源：[1][2][3]。','small')
p.finish()

# 3. MLP quantitative results.
p=Page(3,'结果 / MLP 容量','加宽有收益，参数更多并不保证更好','PSNR、SSIM、MS-SSIM 越高越好；LPIPS 越低越好。主表来自 best 快照渲染评估。')
table=[['配置','MLP 参数','PSNR','ΔPSNR','SSIM','LPIPS','MS-SSIM']]
for r in mlp_rows:
    m=r['metrics'];table.append([r['id'],f"{r['mlp_parameters']:,}",f"{m['PSNR']:.3f}",f"{m['PSNR']-base['PSNR']:+.3f}",f"{m['SSIM']:.6f}",f"{m['LPIPS']:.6f}",f"{m['MS-SSIM']:.6f}"])
p.table(table,[82,69,60,61,78,78,CW-428],highlight=2)
p.image(mlp_chart,165,'图 1  参数量与 PSNR 的关系及 LPIPS 对照。左图横轴为对数刻度；连接线仅辅助阅读，不表示拟合定律。')
p.table([['配置','渲染 best 步数','最终 50k 日志 PSNR'],
         *[[r['id'],f"{mlp_best[r['id']]:,}",f"{r['final_50k_logged_psnr']:.3f}"] for r in mlp_rows]], [139,173,CW-312])
p.text('512×0 比原始 256×0 高 1.320 dB；但在宽度 512 下加入两层隐藏层后，PSNR 降低 0.961 dB。更大模型的收益在本次实验中不呈单调关系，也没有跨所有指标占优。')
p.text('附表中的最终 PSNR 来自训练期间的评估日志，主表则为 PNG 渲染后指标，两者口径不同，不能混作同一列。原始对照为历史运行且缺少完整源码哈希；还需同版本、多种子重复才能判断因果和稳定收益。来源：[1][3]。','small')
p.finish()

# 4. MLP images.
p=Page(4,'结果 / MLP 图像','MLP 容量：同一测试视图的 GT 对照','预先选取测试索引 25；相机、GT 和归一化时间在所有模型中一致。')
p.text('对应原始图像 '+original_test['frames'][25]['file_path']+'，时间 t='+f"{original_test['frames'][25]['time']:.3f}"+'。裁剪窗口仅由 GT 前景确定，并对所有模型使用相同窗口与缩放。','small')
draw_visual(p,mlp_visual,'图 2  GT 与四种 MLP 配置。面板标注各自 best 快照步数；末尾空白仅为网格排版，不代表缺失实验。')
p.text('图像用于比较鸭体轮廓、蓝色支架边缘和局部细节。单帧外观差异不能替代全 50 张测试图像的平均结果，也不能由静态图判断时间一致性。')
p.callout('本次 PSNR 最优候选是 512×0；原始 256×0 的 LPIPS 更低。模型选择应结合目标指标，不能仅按参数量排序。完整 render 与 GT 的绝对地址见第 10 页。')
p.finish()

# 5. Dropout: unchanged official test split.
p=Page(5,'结果 / 固定测试集','随机缺帧：90% 时性能明显下降','所有行使用同一组 50 张官方测试图像；0% 为复用原始 baseline。')
table=[['缺帧','保留帧','PSNR','ΔPSNR','SSIM','LPIPS','MS-SSIM']]
for r in mask_rows:
    m=r['official_metrics'];table.append([f"{r['drop_percent']}%",r['retained'],f"{m['PSNR']:.3f}",f"{m['PSNR']-base['PSNR']:+.3f}",f"{m['SSIM']:.6f}",f"{m['LPIPS']:.6f}",f"{m['MS-SSIM']:.6f}"])
p.table(table,[53,48,68,67,90,90,CW-416],highlight=2)
p.image(mask_chart,170,'图 3  固定官方测试集上的缺帧比例曲线。虚线为 0% 对照；无重复试验，故不绘制误差条。')
p.text('10% 缺帧 PSNR 比复用 0% 对照高 0.996 dB，LPIPS 也略低；该单次结果不能证明删图一定改善模型。30%、50%、70% 的 PSNR 存在波动，而 SSIM、LPIPS 等指标整体显示重建受损。')
p.text('90% 仅保留 15 张训练图像，官方测试 PSNR 为 23.063 dB，比 0% 低 4.194 dB；LPIPS 从 0.029624 升至 0.060427。相对 10% 缺帧，本轮 PSNR 下降约 5.190 dB。')
p.text('跨比例性能比较应以这组固定测试视图为主。变化同时包含数据稀疏、每图采样次数及 baseline 原生 train_count 归一化的影响。来源：[2][4]。','small')
p.finish()

# 6. Dropped-frame GT evaluation.
p=Page(6,'结果 / 被移除帧','被移除训练帧的重建表现','这些图像不参与训练或 best 快照选择；数量和成员随缺帧比例变化。')
table=[['缺帧','评估帧数','PSNR','SSIM','LPIPS','MS-SSIM']]
for r in mask_rows[1:]:
    m=r['masked_holdout_metrics'];table.append([f"{r['drop_percent']}%",r['removed'],f"{m['PSNR']:.3f}",f"{m['SSIM']:.6f}",f"{m['LPIPS']:.6f}",f"{m['MS-SSIM']:.6f}"])
p.table(table,[61,88,86,91,91,CW-417])
p.image(heldout_chart,160,'图 4  两组视图上的 PSNR。两条曲线来自不同图像集合，绝对值不应直接解释为同一难度任务的优劣。')
p.text('10% 缺帧保留集的 PSNR 为 32.994 dB，90% 为 23.521 dB；严重缺帧时，官方测试与被移除帧的重建均明显变差。50% 相比 30% 的小幅回升不能解读为更强泛化，因为集合成员和运行轨迹均不同。')
p.table([['缺帧比例','用于两组渲染的 best 步数'],*[[f"{r['drop_percent']}%",f"{mask_best[r['drop_percent']]:,}"] for r in mask_rows[1:]]], [165,CW-165])
p.text('0% 没有被移除帧，因此该组指标为不适用，并未填零。第 9 页按源图像 ID 对齐一个共同被移除帧，避免把不同图像索引误作同一相机。来源：[2][4]。','small')
p.finish()

# 7-8. Official-image comparisons.
for number,path,title,caption in [
    (7,low_visual,'缺帧图像对照：0%、10%、30%','图 5  固定官方测试索引 25。GT、0%、10%、30% 使用相同裁剪窗口。'),
    (8,high_visual,'缺帧图像对照：50%、70%、90%','图 6  同一官方测试视图。GT、50%、70%、90% 使用与上一页相同的 GT 窗口。')]:
    p=Page(number,'结果 / 官方测试图像',title,'原始图像 '+original_test['frames'][25]['file_path']+' | t='+f"{original_test['frames'][25]['time']:.3f}")
    p.text('仅对展示图像裁剪和等比例缩放，没有颜色增强；指标仍由完整 1600×900 渲染图像计算。','small')
    draw_visual(p,path,caption,450)
    p.text('可对照鸭体轮廓、遮挡关系、支架边缘和颜色细节。这里展示的是一个固定中间视图，不能代替全测试集平均指标；不同配置的快照步数见第 6 页及原始迭代记录。')
    p.text('完整官方测试 render 的绝对地址见第 11 页；0% 原始 baseline 的地址见第 10 页第一项。GT 与 render 各 50 张，文件名逐一对应。','small')
    p.finish()

# 9. A common withheld frame matched by source ID.
p=Page(9,'结果 / 共同被移除帧','同一被移除帧：五档缺帧比例对照',
       '源图像 '+common_id+' | 保留原时间 t='+f"{common_frame['time']:.3f}"+' | 各比例均未用于训练')
p.text('该帧从 10% 缺帧清单的中间位置预先选取。由于缺帧集合嵌套，它同时存在于全部五个被移除集合中。按源 ID 与位姿对齐，GT 像素已验证一致。','small')
draw_visual(p,heldout_visual,'图 7  GT 与五个缺帧模型。面板 index 为各自 heldout PNG 索引；索引不同，但源图像、相机和时间相同。')
p.table([['缺帧比例','对应 heldout PNG'],*[[f'{ratio}%',f'{index:05d}.png'] for ratio,index in heldout_indices.items()]], [165,CW-165])
p.text('该共同视图提供一致的视觉检查对象；各比例的整体被移除帧平均指标仍来自不同集合。所有 heldout render 的绝对地址见第 11 页。','small')
p.finish()

# 10. Absolute MLP image directories.
p=Page(10,'文件索引 / 绝对地址','MLP 实验 render 与 GT 保存位置','以下均为已存在的 Windows 绝对路径。路径换行仅为排版，目录名未省略。')
p.text('图片名为 00000.png 至 00049.png。render 与 GT 文件名对应，分辨率 1600×900；每个目录含 50 张 PNG。','small')
for entry in [x for x in render_catalog if x['family']=='mlp']:
    p.heading(entry['label']+' | best '+f"{entry['best_iteration']:,}"+' | 50 帧')
    p.text('Render：','small',gap=2);p.text(path_markup(entry['render_absolute']),'path',gap=8)
    p.text('GT：','small',gap=2);p.text(path_markup(entry['gt_absolute']),'path',gap=12)
p.text('W256_D0 是复用原始 baseline，也是缺帧实验 0% 的对照。其余三个目录为本轮容量实验独立输出。','small')
p.finish()

# 11. Every missing-frame render directory, with actual counts.
p=Page(11,'文件索引 / 绝对地址','缺帧实验 render 的完整绝对地址','官方测试与被移除帧的输出分开保存；所有目录已核验存在，未用模板替代。')
for r in mask_rows[1:]:
    entries=[x for x in render_catalog if x['family']=='mask' and x['label']==str(r['drop_percent'])+'%']
    p.heading(f"缺帧 {r['drop_percent']}% | best {mask_best[r['drop_percent']]:,}")
    for entry in entries:
        label='官方测试 render' if entry['split']=='official_test' else '被移除帧 render'
        p.text(label+f"（{entry['frame_count']} 张）：",'small',gap=2)
        p.text(path_markup(entry['render_absolute']),'path',gap=8)
p.text('GT 位于每个对应 ours_best 目录下的 gt 子目录，与 renders 同级且文件名一一对应。完整路径及核验清单同时保存在报告的 sources.json 中。','small')
p.finish()

# 12. Interpretation and sources.
p=Page(12,'结论 / 可复核来源','结论、限制与复现入口','本报告只读取已保存的实验记录与渲染图像，没有追加训练或改写结果。')
p.text('<b>MLP 容量结论：</b>在本次运行中，加宽到 512×0 有较好的 PSNR；加深并叠加加宽没有继续提高 PSNR。最优容量应由重建指标和计算成本共同决定，不能假设越大越好。')
p.text('<b>缺帧结论：</b>baseline 在少量缺帧时仍能重建，但 90% 缺帧时两组评估明显恶化。10% 高于历史完整数据对照属于观察结果，仍需同环境、多训练种子与多 mask 种子验证。')
p.text('<b>解释边界：</b>高缺帧增加每张图像的重复次数，并改变原生 fid/train_count 查询。90% 保留帧只覆盖 t=0.067 至 0.919，官方测试包含区间外时刻，故结果也含时间外推影响。规范点云仍共享，不能将此实验解释为完全无几何先验的缺图重建。')
p.text('<b>快照与基线：</b>best 由官方测试 PSNR 选择；被移除帧不参与选择。复用 baseline 缺少完整历史源码哈希，且只有一个场景、一个训练种子与一个缺帧种子，未提供统计显著性或物理参数恢复结论。')
p.heading('验证与执行记录')
p.text('MLP：四种网络均通过形状、参数量、有限梯度检查和短训练 / 渲染 / 评估。缺帧：13 项 CPU 划分与队列测试通过，90% 缺帧短测及五档长训练的两组渲染 / 评估均完成。新队列在 MLP 产物及进程完成后才启动。','small')
p.heading('本地来源')
for line in [
    '[1] experiments/mlp_capacity/results.json、plan.json、validation.json；variants/mlp_capacity/run.py。',
    '[2] experiments/random_frame_mask/results.json、plan.json、preflight.json、source_freeze.json；variants/random_frame_mask/{prepare.py,run.py}。',
    '[3] runs/pidg_mlp_capacity/long_*/rubber_duck/reproduction.json、results.json、train.log；runs/pidg_official/rubber_duck/ 对应记录。',
    '[4] runs/pidg_random_frame_mask/long_drop*_seed0/rubber_duck/reproduction.json、results.json 及 masked_holdout/results.json。',
    '[5] data/pidg_random_frame_mask/rubber_duck/drop*_seed0/split_manifest.json 与 transforms 清单；third_party/PIDG 的原始网络、训练、渲染与指标脚本。']:
    p.text(escape(line),'small',gap=6)
p.text('报告生成：scripts/build_pidg_baseline_ablation_report.py。配套 sources.json 冻结源结果、render 绝对地址、快照步数和图像哈希，便于后续复核。','small')
p.finish()

if split_results:
    split_rows = split_results['rows']
    def split_value(row, key, metric='PSNR'):
        values = row['splits'][key]['metrics']
        return 'N/A' if values is None else f'{values[metric]:.3f}' if metric == 'PSNR' else f'{values[metric]:.6f}'

    p=Page(13,'补充评估 / 指标范围','缺失与保留：PSNR 评估范围说明',
           '复用相同 best 快照；新增评估，不重新训练。保留帧属于已见数据。')
    p.text('<b>原报告的主 PSNR 不是整个数据集的 PSNR。</b>它是固定 50 张官方测试图像的逐图均值；被移除训练帧另行评估。此前没有计算保留训练帧的平均指标。')
    p.table([['集合','帧数与定义'],['保留训练帧','150 × (1 - 缺帧比例)，训练中可见'],
             ['被移除训练帧','150 × 缺帧比例，训练与快照选择中均不可见'],
             ['原训练集全量','保留 + 被移除 = 150 帧，所有比例覆盖同一原始清单'],
             ['完整数据集','原训练 150 + 官方测试 50 = 200 帧，所有比例覆盖相同帧集合']], [112,CW-112])
    p.heading('同一快照的五种统计范围 | PSNR (dB)')
    p.table([['缺帧','保留帧','缺失帧','原训练150','官方测试50','全部200'],
             *[[f"{r['drop_percent']}%",*[split_value(r,k) for k in ('retained','removed','original_train','official_test','all_200')]] for r in split_rows]],
            [49,83,83,96,100,CW-411])
    p.text('<b>合并方法：</b>先对每张图计算指标，再对所有目标图像取算术平均。原训练均值 = (保留帧数 × 保留均值 + 缺失帧数 × 缺失均值) / 150；全量均值 = (150 × 原训练均值 + 50 × 官方测试均值) / 200。')
    p.text('该 PSNR 口径与原 PIDG 一致：平均逐图的 dB 值，而不是把全部图像像素合在一起先求 MSE 再转为 dB。SSIM、LPIPS、MS-SSIM 同样按帧加权。0% 没有缺失帧，记作 N/A。','small')
    p.text('完整 200 帧含训练中已见图像，因此属于全量重建统计；泛化能力仍应参考固定官方测试和被移除帧。不同缺帧比例的保留 / 缺失集合不同，原训练 150 帧和全部 200 帧则是相同视图集合。','small')
    p.finish()

    p=Page(14,'补充评估 / 分组指标','保留训练帧与被移除帧：完整指标',
           'PSNR、SSIM、MS-SSIM 越高越好；LPIPS 越低越好。每个比例使用原先相同的 best 快照。')
    p.table([['缺帧','子集','帧数','PSNR','SSIM','LPIPS','MS-SSIM'],
             *[[f"{r['drop_percent']}%",title,r['splits'][key]['count'],
                *[split_value(r,key,m) for m in ('PSNR','SSIM','LPIPS','MS-SSIM')]]
               for r in split_rows for key,title in [('retained','保留 / 已见'),('removed','缺失 / 未见')]]],
            [43,89,37,71,89,89,CW-418])
    p.text('每行均由真实逐帧记录汇总，不使用另一集合的图像替代；输入源 ID、相机位姿和时间已经对齐。较高缺帧比例的缺失集合包含较低比例已缺失的帧。')
    p.text('保留部分是训练数据上的重建表现，被移除部分是同场景未见训练帧上的重建表现；二者采样集合不同，数值差异不能单独归因为过拟合。背景、原始 GT 与 1600×900 分辨率均沿用官方渲染 / 指标协议。','small')
    p.text('新增保留帧仅用于本次训练后评估，不回传梯度、不更新权重、不改变 best 快照。新增输出与原官方测试、masked_holdout 目录分开保存。','small')
    p.finish()

    p=Page(15,'补充评估 / 合并指标','原训练集、官方测试与全量：完整指标',
           '原训练 150 与全部 200 的帧集合在六档缺帧比例间相同。合并均值按真实帧数计算。')
    p.table([['缺帧','集合','帧数','PSNR','SSIM','LPIPS','MS-SSIM'],
             *[[f"{r['drop_percent']}%",title,r['splits'][key]['count'],
                *[split_value(r,key,m) for m in ('PSNR','SSIM','LPIPS','MS-SSIM')]]
               for r in split_rows for key,title in [('original_train','原训练'),('official_test','官方测试'),('all_200','全部数据')]]],
            [43,74,38,74,93,93,CW-415], size='small')
    p.text('以上统计复用逐帧官方测试与缺失帧结果，并补算 525 张保留帧图像。模型和原始数据哈希保持一致；采用原 PIDG 指标函数与 VGG LPIPS，旧官方 PNG 的四项指标一致性检查通过。新增 16 项测试通过，全部 1,200 对模型 / 图像的 GT 源身份与 PSNR 已独立核验。','small')
    p.text('来源：experiments/random_frame_mask/split_metrics.json；同目录 split_metrics.md 与 split_metrics.csv 提供相同数值。各运行 retained_evaluation/all_frame_metrics.json 保存全部 200 帧的源 ID、分组、图像地址与逐帧指标。','small')
    p.finish()

    p=Page(16,'补充评估 / 绝对地址','新增保留训练帧 render 的保存位置',
           '以下六项均为已存在的完整 Windows 绝对路径；换行仅为排版。')
    for r in split_rows:
        p.heading(f"缺帧 {r['drop_percent']}% | 保留 {r['splits']['retained']['count']} 帧 | best {r['best_iteration']:,}")
        p.text(path_markup(r['retained_render_absolute']),'path',gap=13)
    p.text('每项 renders 目录的同级 gt 目录保存对应 GT，文件名从 00000.png 开始，数量与该比例的保留帧数一致。完整 GT 绝对地址同时写入 split_metrics.json 和本报告 sources.json。','small')
    p.text('被移除帧的 render 地址见第 11 页；0% 官方测试地址见第 10 页，其余官方测试地址见第 11 页。本次只新增保留帧输出，既有渲染和模型快照均已保留。','small')
    p.text('复现入口：variants/random_frame_mask/evaluate_splits.ps1。它配置现有 CUDA 环境，运行只读快照评估；已有完整输出可复用。','small')
    p.finish()
c.save()

source_paths=[ROOT/'experiments/mlp_capacity/results.json',ROOT/'experiments/mlp_capacity/plan.json',
              ROOT/'experiments/mlp_capacity/validation.json',ROOT/'experiments/random_frame_mask/results.json',
              ROOT/'experiments/random_frame_mask/plan.json',ROOT/'experiments/random_frame_mask/preflight.json',
              ROOT/'variants/mlp_capacity/run.py',ROOT/'variants/random_frame_mask/run.py',
              ROOT/'variants/random_frame_mask/prepare.py',Path(__file__)]
archive={'pdf':str(PDF),'created_london':REPORT_TIME.isoformat(),'pages':page_records,
         'mlp_snapshot':mlp,'random_frame_mask_snapshot':mask,'render_catalog':render_catalog,
         'total_render_frames':sum(x['frame_count'] for x in render_catalog),
         'run_records':run_records,'image_checks':image_checks,
         'common_heldout_frame':common_frame,'heldout_png_indices':heldout_indices,
         'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in source_paths}}
if split_results:
    archive['split_evaluation_snapshot'] = split_results
    archive['split_evaluation_validation'] = split_validation
    archive['source_sha256'][str(split_path.relative_to(ROOT))] = digest(split_path)
    validation_path = ROOT / 'experiments/random_frame_mask/split_metrics_validation.json'
    archive['source_sha256'][str(validation_path.relative_to(ROOT))] = digest(validation_path)
    evaluator_path = ROOT / 'variants/random_frame_mask/evaluate_splits.py'
    archive['source_sha256'][str(evaluator_path.relative_to(ROOT))] = digest(evaluator_path)
    validator_path = ROOT / 'variants/random_frame_mask/validate_splits.py'
    archive['source_sha256'][str(validator_path.relative_to(ROOT))] = digest(validator_path)
(OUT/'PIDG_MLP_MaskedData_Report.sources.json').write_text(json.dumps(archive,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(PDF),'pages':len(page_records),'render_directories':len(render_catalog),
                  'render_frames':archive['total_render_frames'],'common_heldout_id':common_id,
                  'content_bottom':[x['content_bottom'] for x in page_records]},ensure_ascii=False))

