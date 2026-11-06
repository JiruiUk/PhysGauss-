"""Create a dated, evidence-snapshotted Chinese weekly experiment summary.

Read-only with respect to training code, models, experiments and old reports.
Run with the bundled Codex Python runtime. Only this report and its QA files
are authored; no training or inference is started.
"""
from pathlib import Path
from datetime import datetime, timezone
from html import escape
import hashlib
import json
import subprocess
import re
import shutil
from weekly_image_atlas_20261103 import prepare_atlas, render_atlas, ATLAS_PAGE_COUNT
from weekly_s6_update_20261103 import prepare_s6, render_s6, S6_PAGE_COUNT
from weekly_test_comparison_20261103 import prepare_comparison, render_comparison, COMPARE_PAGE_COUNT

from PIL import Image as PILImage
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
BASE = 'PIDG_Weekly_Experiment_Summary_CN_20261103'
OUT = ROOT / 'output/pdf'
TMP = ROOT / 'tmp/pdfs/weekly_20261103'
SNAP = TMP / 'evidence'
PDF = OUT / (BASE + '.pdf')
TOTAL_PAGES = 17 + ATLAS_PAGE_COUNT + S6_PAGE_COUNT + COMPARE_PAGE_COUNT
OUT.mkdir(parents=True, exist_ok=True)
SNAP.mkdir(parents=True, exist_ok=True)

# Stable snapshot: later work in other threads must not silently change this report.
REFS = [
 ('时间划分、共同帧时钟与各向同性对照', 'experiments/kinematic_resplit/results.md'),
 ('覆盖、邻域与增密 C0-C5', 'output/pdf/PIDG_Coverage_Ablation_Report_CN.md'),
 ('颜色梯度修正 C1-C3', 'output/pdf/PIDG_Color_Repair_Ablation_Report_CN.md'),
 ('全局质心 G0/G1', 'experiments/centroid_tuning/assessment.md'),
 ('完整度联合消融 H1-H4', 'experiments/integrity_tuning/assessment.md'),
 ('完整训练与反馈 F1/F2', 'experiments/integrity_continuation/assessment.md'),
 ('形变修正 D1-D3', 'experiments/deformation_repair/assessment.md'),
 ('11月1日全模块复核，含R1/R2状态', 'output/research_audit_20261101/RESEARCH_AUDIT_CN.md'),
 ('固定/可学共享刚度', 'experiments/c1_material/assessment.md'),
 ('冻结/可学物体分组', 'experiments/object_id/assessment.md'),
 ('冻结轨迹的速度拟合与时钟碰撞', 'experiments/frozen_velocity_probe/assessment.md'),
 ('单调时钟联合训练 J1', 'experiments/monotonic_training/assessment.md'),
 ('速度来源 K1/K2 与光流流程审计', 'experiments/flow_comparison/assessment.md'),
 ('修正光流与共享应力 L0/L1', 'experiments/unified_flow/assessment.md'),
 ('动量梯度保护 M1', 'experiments/momentum_repair/assessment.md'),
 ('统一离散轨迹动量 N1', 'experiments/coherent_momentum/assessment.md'),
 ('C1光流开关消融', 'experiments/c1_flow_ablation/assessment.md'),
 ('局部形态候选P1的准入失败', 'experiments/shape_coherence/assessment.md'),
 ('全时段早期点库支持审计', 'experiments/surface_support/assessment.md'),
 ('运动历史与遮挡歧义', 'experiments/motion_correspondence/assessment.md'),
 ('16个局部可见表面点库', 'experiments/surface_correspondence/assessment.md'),
 ('清晰短片段筛查与尺度复核', 'experiments/short_clip_diagnostic/assessment.md'),
 ('物体-点库关联链', 'experiments/object_membership/assessment.md'),
 ('关联链续训 T0/T1', 'experiments/association_training/assessment.md'),
 ('固定相机全时间审计', 'experiments/fixed_camera_temporal/assessment.md'),
 ('无GT完整度代理指标', 'experiments/view_completeness/assessment.md'),
 ('首版几何迁移续训 U0/U1', 'experiments/transfer_integrity/assessment.md'),
 ('从零共同预热对照', 'experiments/scratch_integrity/assessment.md'),
 ('硬静态点池 R0', 'experiments/static_pool/assessment.md'),
 ('静态参考准备审计 R0a', 'experiments/static_reference_audit/assessment.md'),
 ('独立静态重建 Stage S1', 'experiments/staged_reconstruction/S1_assessment.md'),
 ('Stage S2首版与两次修正', 'experiments/staged_reconstruction/S2_geometry_recovery/assessment.md'),
 ('Stage S3物理场接入', 'experiments/staged_reconstruction/S3/assessment.md'),
 ('Stage S4稀疏轨迹接入', 'experiments/staged_reconstruction/S4/assessment.md'),
 ('Stage S5条件内部先验', 'experiments/staged_reconstruction/S5/assessment.md'),
 ('Stage S1-S6执行计划，11:07历史快照', 'experiments/staged_reconstruction_plan.md'),
 ('S6正式终态评估与图像复核', 'experiments/staged_reconstruction/S6/assessment.md'),
 ('S1-S6全链实测与最终状态', 'experiments/staged_reconstruction/final_summary.md'),
]
EXTRA = [
 'experiments/v3_iso_scratch/results.md',
 'experiments/v3_isotropic/results.md',
 'experiments/staged_reconstruction/S2_assessment.md',
 'experiments/staged_reconstruction/S2_recovery/assessment.md',
 'experiments/staged_reconstruction/S6/smoke_verification.json',
 'experiments/scratch_integrity/numeric_summary.json',
 'experiments/static_pool/final_audit.json',
 'experiments/short_clip_audit/assessment.md',
 'experiments/staged_reconstruction/S5/S5_val_04000.json',
 'experiments/staged_reconstruction/S5/S5_test_04000.json',
]
FIGURES = [
 'output/fixed_camera_temporal/key_comparison.png',
 'output/scratch_integrity/native_inspection_fixed86.png',
 'output/scratch_integrity/native_crop_fixed86_t2.png',
 'output/static_pool/review/fixed_0086_native_crops.png',
 'output/staged_reconstruction/S1_static/S1_review.png',
]
manifest_path = TMP / 'snapshot.json'
if not manifest_path.exists():
    stamp = datetime.now(timezone.utc).isoformat()
    sources = {}
    for rel in [r[1] for r in REFS] + EXTRA + FIGURES:
        p = ROOT / rel
        if not p.exists():
            if rel in EXTRA or rel in FIGURES:
                continue
            raise FileNotFoundError(p)
        data = p.read_bytes()
        target = SNAP / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        sources[rel] = {'sha256': hashlib.sha256(data).hexdigest(),
                        'size': len(data), 'mtime_utc': datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()}
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest_path.write_text(json.dumps({'cutoff_utc': stamp, 'git_head': head, 'sources': sources}, ensure_ascii=False, indent=2), encoding='utf-8')
MANIFEST = json.loads(manifest_path.read_text(encoding='utf-8'))
# Supplement two older, read-only 10/26 result records discovered in the final
# coverage audit; do not refresh live stage files or change the report cutoff.
for rel in ('experiments/v3_iso_scratch/results.md', 'experiments/v3_isotropic/results.md'):
    if rel not in MANIFEST['sources']:
        src = ROOT / rel
        data = src.read_bytes()
        target = SNAP / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        MANIFEST['sources'][rel] = {'sha256': hashlib.sha256(data).hexdigest(),
            'size':len(data), 'mtime_utc':datetime.fromtimestamp(src.stat().st_mtime,timezone.utc).isoformat(),
            'supplemental_read_utc':datetime.now(timezone.utc).isoformat(),
            'historical_record_updated_utc':'2026-10-26'}
manifest_path.write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2),encoding='utf-8')
CUTOFF = datetime.fromisoformat(MANIFEST['cutoff_utc']).strftime('%Y-%m-%d %H:%M UTC')
old_report=TMP/(BASE+'.original_17pages.pdf')
if PDF.exists() and not old_report.exists():
    shutil.copy2(PDF,old_report)
previous_image_report=TMP/(BASE+'.before_S6_32pages.pdf')
if PDF.exists() and not previous_image_report.exists():
    shutil.copy2(PDF,previous_image_report)
previous_s6_report=TMP/(BASE+'.before_test_comparison_36pages.pdf')
if PDF.exists() and not previous_s6_report.exists():
    shutil.copy2(PDF,previous_s6_report)
ATLAS,ATLAS_MANIFEST=prepare_atlas(ROOT,SNAP)
S6_ASSETS,S6_MANIFEST,S6_DATA=prepare_s6(ROOT,SNAP)
COMPARE_ASSETS,COMPARE_DATA=prepare_comparison(ROOT,SNAP)
S6_CUTOFF=datetime.fromisoformat(S6_MANIFEST['captured_utc']).strftime('%Y-%m-%d %H:%M UTC')
MANIFEST['s6_revision']={'captured_utc':S6_MANIFEST['captured_utc'],
    'checkpoint_sha256':S6_MANIFEST['checkpoint_sha256'],
    'source_manifest':BASE+'.sources.s6.json',
    'scope':'formal S6 results supplement; historical source snapshots preserved'}
MANIFEST['test_comparison_revision']={'captured_utc':COMPARE_DATA['captured_utc'],
    'baseline':'H3_combined final50k','models':['S0_scratch_control','Stage S1','S2','S3','S4','S5','S6'],
    'test_frames':50,'source_file':BASE+'.test_comparison.json',
    'new_training_updates':0,'scope':'common-GT frozen-model evaluation; original historical metrics preserved'}
MANIFEST['image_revision']={'page_count':ATLAS_PAGE_COUNT,
    'source_count':len(ATLAS_MANIFEST['sources']),
    'manifest':'tmp/pdfs/weekly_20261103/evidence/atlas/manifest.json',
    'prepared_utc':ATLAS_MANIFEST['prepared_utc']}

pdfmetrics.registerFont(TTFont('Deng', 'C:/Windows/Fonts/Deng.ttf'))
pdfmetrics.registerFont(TTFont('DengB', 'C:/Windows/Fonts/Dengb.ttf'))
pdfmetrics.registerFontFamily('Deng', normal='Deng', bold='DengB', italic='Deng', boldItalic='DengB')
W, H = A4
M, CW = 46, W - 92
INK = colors.HexColor('#1B2B3C')
NAVY = colors.HexColor('#16364D')
TEAL = colors.HexColor('#087C83')
GRAY = colors.HexColor('#596A78')
LIGHT = colors.HexColor('#EEF4F6')
RULE = colors.HexColor('#D9E3E8')
ORANGE = colors.HexColor('#A65B25')
styles = {
 'body': ParagraphStyle('body', fontName='Deng', fontSize=10.3, leading=16, textColor=INK, wordWrap='CJK'),
 'small': ParagraphStyle('small', fontName='Deng', fontSize=8.6, leading=12.9, textColor=GRAY, wordWrap='CJK'),
 'cell': ParagraphStyle('cell', fontName='Deng', fontSize=9.0, leading=13.3, textColor=INK, wordWrap='CJK'),
 'head': ParagraphStyle('head', fontName='DengB', fontSize=8.8, leading=12.5, textColor=colors.white, wordWrap='CJK'),
 'callout': ParagraphStyle('callout', fontName='DengB', fontSize=11.3, leading=18, textColor=NAVY, wordWrap='CJK'),
 'source': ParagraphStyle('source', fontName='Deng', fontSize=8.0, leading=11.4, textColor=GRAY, wordWrap='CJK'),
}

c = canvas.Canvas(str(PDF), pagesize=A4, pageCompression=1)
c.setTitle('PIDG 上周与本周实验总结 | 2026.10.26 - 2026.11.03')
c.setAuthor('PhysGauss / PIDG experiment records')
c.setSubject('两周实验设置、完成状态、正负结果、固定视角诊断及后续方向')
Y = 0
PAGE = 0
QA = []
TRANSCRIPT = []

def clean(s):
    return str(s).replace('\u2011','-').replace('\u2013','-').replace('\u2014','-').replace('\u2212','-')

def fmt(s):
    s = escape(clean(s))
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s).replace('\n','<br/>')

def check(y):
    if y < 47:
        raise ValueError(f'Page {PAGE} overflow: y={y:.1f}')

def p(s, kind='body', gap=9):
    global Y
    ob = Paragraph(fmt(s), styles[kind])
    # Reserve a small right gutter for CJK punctuation and mixed Latin text.
    _, hh = ob.wrap(CW-12, H)
    check(Y - hh)
    ob.drawOn(c, M, Y-hh)
    Y -= hh + gap
    TRANSCRIPT.append(clean(s))

def heading(s):
    global Y
    Y -= 7
    check(Y-19)
    c.setFillColor(TEAL); c.setFont('DengB',12)
    c.drawString(M,Y-12,clean(s))
    Y -= 24
    TRANSCRIPT.append(clean(s))

def box(s):
    global Y
    ob=Paragraph(fmt(s),styles['callout'])
    _, hh=ob.wrap(CW-28,H)
    height=hh+23
    check(Y-height)
    c.setFillColor(LIGHT); c.roundRect(M,Y-height,CW,height,5,stroke=0,fill=1)
    c.setFillColor(TEAL); c.rect(M,Y-height,3,height,stroke=0,fill=1)
    ob.drawOn(c,M+14,Y-12-hh)
    Y -= height+15
    TRANSCRIPT.append(clean(s))

def table(headers, rows, widths=None, highlights=()):
    global Y
    if widths is None: widths=[CW/len(headers)]*len(headers)
    else: widths=[CW*v/sum(widths) for v in widths]
    data=[[Paragraph(fmt(s), styles['head']) for s in headers]]
    data += [[Paragraph(fmt(s),styles['cell']) for s in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT')
    commands=[('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
              ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
              ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
              ('LINEBELOW',(0,0),(-1,0),.5,NAVY),('LINEBELOW',(0,1),(-1,-1),.35,RULE)]
    for i in range(2,len(data),2): commands.append(('BACKGROUND',(0,i),(-1,i),colors.HexColor('#F5F8FA')))
    for i in highlights: commands.append(('BACKGROUND',(0,i+1),(-1,i+1),colors.HexColor('#E0F0ED')))
    t.setStyle(TableStyle(commands))
    _,hh=t.wrap(CW,H)
    check(Y-hh)
    t.drawOn(c,M,Y-hh)
    Y-=hh+13
    TRANSCRIPT.append(' | '.join(headers))
    TRANSCRIPT.extend(' | '.join(map(str,row)) for row in rows)

def compact_comparison_table(headers,rows):
    global Y
    cellstyle=ParagraphStyle('testcell',fontName='Deng',fontSize=7.9,leading=9.6,textColor=INK,wordWrap='CJK',alignment=1)
    headstyle=ParagraphStyle('testhead',fontName='DengB',fontSize=8,leading=9.6,textColor=colors.white,alignment=1)
    data=[[Paragraph(fmt(s),headstyle) for s in headers]]
    data += [[Paragraph(fmt(s),cellstyle) for s in row] for row in rows]
    weights=[.55]+[1]*8
    t=Table(data,colWidths=[CW*v/sum(weights) for v in weights])
    commands=[('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),
        ('TOPPADDING',(0,0),(-1,-1),2.7),('BOTTOMPADDING',(0,0),(-1,-1),2.7),
        ('LINEBELOW',(0,0),(-1,0),.5,NAVY),('LINEBELOW',(0,1),(-1,-1),.3,RULE)]
    for i in range(2,len(data),2):commands.append(('BACKGROUND',(0,i),(-1,i),colors.HexColor('#F1F6F7')))
    t.setStyle(TableStyle(commands));_,height=t.wrap(CW,H);check(Y-height)
    t.drawOn(c,M,Y-height);Y-=height+4
    TRANSCRIPT.append(' | '.join(headers));TRANSCRIPT.extend(' | '.join(row) for row in rows)

def fig(rel, max_h, caption):
    global Y
    src=SNAP/rel
    if not src.exists(): raise FileNotFoundError(src)
    with PILImage.open(src) as im:
        iw,ih=im.size
        ratio=min(CW/iw,max_h/ih)
        ww,hh=iw*ratio,ih*ratio
        check(Y-hh)
        c.drawImage(ImageReader(im.convert('RGB')),M+(CW-ww)/2,Y-hh,ww,hh)
    Y-=hh+6
    p(caption,'small',12)

def start(kicker,title,sub=''):
    global Y,PAGE
    if PAGE:
        QA.append({'page': PAGE,'bottom_content_y': round(Y,2)})
        c.showPage()
    PAGE+=1
    c.setFillColor(TEAL); c.rect(M,H-38,26,3,fill=1,stroke=0)
    c.setFont('Deng',8); c.setFillColor(GRAY)
    c.drawString(M+34,H-38,clean(kicker))
    c.drawRightString(W-M,H-38,'PIDG / 两周实验记录')
    c.setFont('DengB',21); c.setFillColor(NAVY)
    c.drawString(M,H-79,clean(title))
    c.bookmarkPage(f'p{PAGE}')
    c.addOutlineEntry(clean(title),f'p{PAGE}',0,False)
    Y=H-99
    if sub: p(sub,'small',15)
    c.setStrokeColor(RULE); c.setLineWidth(.5); c.line(M,37,W-M,37)
    c.setFont('Deng',7.5); c.setFillColor(GRAY)
    c.drawString(M,24,'2026.10.26 - 2026.11.03  |  Rubber Duck')
    c.drawRightString(W-M,24,f'{PAGE:02d} / {TOTAL_PAGES}')
    TRANSCRIPT.extend(['\n\f',f'{PAGE:02d} {title}',sub])

start('01 / 总览','PIDG 两周实验总结',f'从图像拟合，走向持久物体表示\n上周 2026.10.26 - 11.01  ·  本周 2026.11.02 - 11.03\n历史快照：{CUTOFF}；S6正式结果补充快照：{S6_CUTOFF}。')
box('两周最重要的变化：观测图像可以拟合得很好，但固定相机下仍会碎裂；局部光流或几何损失变好，尚未转化为跨遮挡身份恢复。')
heading('上周：找到较好的图像候选，暴露物理分支问题')
p('颜色修正后的 C1 成为后续常用对照；覆盖与局部应变组合 H3 在旧评估口径下将测试 PSNR 从 40.3721 提升至 40.8782 dB。共享刚度、软物体分组和多种后续正则没有形成全面改进。冻结速度拟合发现内部时钟混叠风险。[02-11]')
heading('本周：把“看起来好”拆成可检查的证据')
p('完成光流/速度/共享应力/动量对照，建立局部“图像物体-表面点库-Gaussian 行”关联；固定相机扫描确认原轨迹RGB分数不能代表新视角完整性。S1-S6条件训练及评估已执行，S6改善观测与轨迹拟合，固定新视角仍碎裂。[12-38]')
table(['代表结果','观测变化','当前判断'],[
 ('H3 / C1，final50k','PSNR +0.5061 dB\n旧LPIPS下降9.55%','保留为上周图像候选'),
 ('从零几何约束 / 匹配对照','短轨迹EPE下降84.5%\n测试PSNR -0.0271 dB','约束生效；固定视角未修复'),
 ('硬静态池 / 匹配对照','静态蓝色贡献51.60%→12.12%','静态子集透明化；不晋级'),
 ('分阶段 Stage S1-S6','S6动态验证28.327 dB\n轨迹较S3下降20.44%','正式4k已评估；几何/身份未解决'),
 ('S6 / 历史H3共同test复评','全幅PSNR -10.501 dB\n动态PSNR +0.268 dB','全幅胜0/50；动态胜27/50'),
 ],[1.05,1.1,1.05],highlights=(0,))
p('阅读索引：实验口径p2；上周p3-5；本周运动与关联p6-8；固定视角与两版训练p9-12；静态与分阶段p13-14；判断p15；历史图像p16-30；S6正式指标与图像p31-34；共同test逐帧对照p35-47；来源p48-49。','small')
p('逐帧更新版：保留历史与S6图像，新增H3参考及S0-Stage S6的50张test逐帧比较：PSNR、SSIM、标准LPIPS及共同动态/静态区域；S1只报告静态。全部精度数值另附CSV。','small')

start('02 / 协议与时间','先明确哪些结果可以比较','报告依据仓库实际记录；按实验完成或验收时间归周，跨周项目单列。')
table(['项目','本报告采用的口径'],[
 ('主要场景','Rubber Duck。其他本地场景未取得本两周同等完整的实验结果，不外推到烟雾、布料或全基准。'),
 ('主要数据划分','200时刻；120训练 / 30验证 / 50历史测试。t=(原帧号-1)/199。每时刻仅一个实际相机观测。'),
 ('图像分辨率','主要正式渲染1600×900。运动EPE多在400×225坐标计算；不可与原生像素直接比较。'),
 ('模型和预算','C1指修复后的C1，并非原始PIDG。final50k、验证选择best、50k→52k续训和15k分叉需分开。'),
 ('LPIPS','早期日志与报告使用旧输入范围；本周统一PNG评估使用VGG及[-1,1]。只在同一表内比较，跨表不排名。'),
 ('完整度与物理量','2D alpha覆盖、细薄支撑、局部轨迹和内部残差分别报告。固定新视角无独立GT；无真实3D速度/应力/材料ID。'),
 ],[.8,3.2])
heading('两周的工作主线')
table(['时间','实验族与进展'],[
 ('10/26-10/29','时间划分重训、各向同性从零对照、覆盖/邻域/增密、颜色修正、质心、固定/可学共享刚度。'),
 ('10/30-11/01','物体分组；H/F完整度与反馈；D/R形变修正；冻结速度拟合。'),
 ('11/01启动，11/02完成','J1单调时钟与运动学一致性联合训练。'),
 ('11/02','K/L/M/N运动与动量对照；C1光流消融；P1准入审查；表面/短片段/关联链诊断与续训。'),
 ('11/03含S6补充','固定视角审计及metric；两版几何训练；静态池失败与审计；Stage S1-S6条件训练及原生评估完成，原始几何/身份/接触目标保留。'),
 ],[1,3])
p('所有主结论仍受单场景、通常单种子、CUDA非确定性与历史测试曝光限制。共同起点不保证GPU逐位一致；多处激活前已有曲线漂移。新增阶段 S1-S6 是重建课程，不等同于 AGENTS.md 的物理里程碑 M0-M10，也不代表完成全部官方基准复现。[01,08,36]','small')

start('03 / 上周 · 重建','覆盖改善与颜色修正','已完成训练：C0-C5、修复后C1-C3、G0/G1。数字均为各自同协议final50k测试结果。')
table(['覆盖组','PSNR / dB','Mask IoU','主要结果'],[
 ('C0 / V3','39.7240','0.2064','RGB高，但背景alpha误覆盖45.60%'),
 ('C1：mask','35.7459','0.9764','轮廓更准，颜色/前景质量明显退化'),
 ('C2：mask+仿射','37.1511','0.9842','弥补部分RGB损失，未超过C0'),
 ('C3：再加增密','35.8634','0.9684','没有稳定叠加收益'),
 ('C4：仅仿射','38.6373','0.2168','局部邻域不等于可靠物体拓扑'),
 ('C5：仅增密','39.3918','0.2360','部分感知指标略好；整体有取舍'),
 ],[1.25,.75,.75,2.15])
p('高alpha的黑斑可能来自颜色或遮挡错误，不会被“透明缺口率”计入。旧C1在困难黄色ROI中红通道被截为零，支持先修颜色梯度，而非只继续补点。[02,03]')
heading('修复后：保留新C1作为图像基线')
table(['修复后模型','PSNR / dB','旧口径LPIPS','相对新C1的判断'],[
 ('C1_repaired','40.3721','0.010438','Softplus颜色 + 调弱mask + 前景RGB'),
 ('C2_repaired','39.2914','0.011191','再加弱仿射；低覆盖改善但RGB退步'),
 ('C3_repaired','39.2738','0.011240','增298点；相对C2无稳定整体收益'),
 ],[1.1,.7,.8,1.8],highlights=(0,))
p('颜色修正是组合干预，不能将全部提升独立归功于Softplus。新C1相对旧C1，PSNR提高4.6262 dB，但Mask IoU从0.9764降至0.9161，背景误覆盖升至1.004%。[03]')
heading('质心约束：平均值提高，不代表每帧更好')
p('G0→G1：PSNR 37.1394→37.5279 dB，质心距离121.52→0.56 px；SSIM与LPIPS变差，仅23/50帧PSNR改善，逐帧差中位数-0.1319 dB。收益主要由最早三帧驱动。全局alpha质心无法约束局部形态、深度或物体身份。[04]')
p('日期范围补充：10/26时间划分重训V3/V4的final PSNR为40.0917/40.1055 dB，运动学项仅+0.0138 dB且感知指标混合；它们与本页C0属于不同训练记录，不互换数值。[01]','small')
heading('各向同性：改用从零对照后，未见全面收益')
p('10/26同一随机step0、每组50k，iso=0→0.001：PSNR 39.7240→39.5581 dB，旧LPIPS 0.0106815→0.0111843；SSIM和MS-SSIM略升。此前续训方案保存停止，对照仅额外10,489步、iso臂1000步，没有最终比较；未将该未完成续训当成负结果。[01]','small')

start('04 / 上周 · 完整度','H3是候选，后续叠加未稳定胜出','相同50张历史测试、final50k；旧口径LPIPS。低覆盖为GT核心区alpha<0.9的百分比。')
table(['配置','PSNR','旧LPIPS','低覆盖%','结论'],[
 ('C1 对照','40.3721','0.010438','0.09076','起点'),
 ('H1 覆盖','40.0027','0.010549','0.08387','覆盖局部改善，画质退步'),
 ('H2 时间应变','39.9132','0.010316','0.09248','未形成全面收益'),
 ('H3 H1+H2','40.8782','0.009441','0.06059','综合候选；33/50帧PSNR提高'),
 ('H4 颜色增密','39.3669','0.010630','0.07898','实际增363点，未带来整体收益'),
 ('F1 困难像素','39.7816','0.010981','0.06543','相对H3退步1.0966 dB'),
 ('F2 减半权重','39.9096','0.010698','0.11533','没有兑现验证收益'),
 ('D1 运动仿射','40.9159','0.009576','0.07537','均值略高，尾部和验证护栏失败'),
 ('D2 协方差曲率','39.9250','0.010677','0.11873','可能误罚合法刚体旋转'),
 ('D3 D1+D2','40.6186','0.009860','0.06482','综合取舍，不替换H3'),
 ],[1.13,.65,.8,.7,1.72],highlights=(3,))
box('H3相对C1：PSNR +0.5061 dB；低覆盖下降33.24%；背景误覆盖1.0041%→0.8036%。这证明观测图像代理改善，尚不证明3D完整。')
p('H3仍有明显坏帧：原帧85的PSNR比C1低6.2382 dB。D1修复了该帧的一部分，但最差5帧均值从H3的34.1705降至33.8594 dB；不能凭测试均值+0.0376 dB取代预设验证选择。[05-08]')
heading('未完成项必须保留原状态')
p('R1四元数单位化已到50k，仅有验证结果，验证PSNR 40.1520 dB；不能补造测试分数。R2困难帧采样在31,900步保存停止；R3、H5未运行。多项新模块激活前已分岔，不能将所有最终差值解释为单项因果效应。[08]')

start('05 / 上周 · 物理与身份','刚度、分组与速度诊断','三条探索线：材料约束是否有益、软分组是否等于物体、独立速度能否拟合渲染运动。')
table(['共享刚度实验','测试PSNR','旧LPIPS','判断'],[
 ('P0 C1匹配对照','40.0373','0.010476','本实验自己的对照，非40.3721那次C1'),
 ('P1 固定E','37.2739','0.015556','48/50帧PSNR更差'),
 ('P2 可学E','37.2673','0.015262','49/50帧更差；E仅1→0.99348'),
 ],[1.2,.75,.8,1.8])
p('局部dL/dF存在大幅尾值，P2最大约187,195；这是局部F导数，不是模型权重梯度。当前全局有效材料先验不作默认改进，不能凭小幅E变化宣布材料识别成功。[09]')
p('物体分组：冻结几何分组O1与可学O2的测试PSNR为38.8108/38.4809 dB，均低于复用C1；可学身份分配较分散，16槽仍有占用。RGB有梯度、槽有使用，并不证明获得真实物体ID。该诊断性物体分组不等同于已知材料分区。[10]')
heading('冻结H3轨迹，单独拟合速度：11/1完成')
table(['速度拟合诊断','拟合误差%','时间留出%','ID+时间留出%'],[
 ('原H3','100.04','100.04','100.11'),
 ('A 小样本','2.16','117.46','148.73'),
 ('B1 原内部时钟','67.29','85.77','106.17'),
 ('B2 较小学习率','70.70','81.00','98.10'),
 ('C 固定单调时钟','20.26','21.97','107.91'),
 ],[1.6,.8,.85,1.05],highlights=(4,))
p('相对RMSE以冻结轨迹导数为teacher，零速度参照为100%。小样本能拟合，排除了完全断梯度；单调时钟大幅改善时间留出，但跨ID迁移仍差。[11]')
p('发现两个相隔很远的时刻被压成几乎相同内部标量时间，而目标速度明显不同。补充单调时钟对照支持时间混叠/条件性是瓶颈之一；它没有使用真实速度GT，也没有产生新的图像质量收益。B3随机初始化诊断训练/时间留出为68.84%/91.73%，不改变上述判断。[11]','small')

start('06 / 本周 · 运动与动量','共享轨迹不自动满足动量方程','J1跨周完成；K/L/M/N均在11/2完成。下表使用各实验验证PSNR选择的best测试快照。')
table(['方案','测试PSNR / dB','独立运动/关键结果'],[
 ('J1 单调时钟+一致性','40.0809','final内部速度偏差仍99.73%；相对H3画质下降'),
 ('K1 独立速度+流','38.3013','验证速度推进EPE 1.1932 px'),
 ('K2 轨迹差分速度+流','34.2065','EPE 0.8468；轨迹EPE反而变差'),
 ('L0 修正流+独立stress','33.9538','速度推进EPE 0.8373 px'),
 ('L1 共享形变latent应力','33.9658','速度推进EPE 0.8363；无综合收益'),
 ('M1 动量共享梯度保护','39.7018','相对L1恢复图像，但动量残差未解决'),
 ('N1 一致离散区间+WLS','39.9095','运动略好；跨时刻动量门槛仍未通过'),
 ],[1.5,.9,2.2])
heading('K轮促成了辅助光流流程修正')
p('实际PIDG加载器的FoV排列与常规元数据约定不同；CUDA的mean-pixel offset符号、alpha合成前后的T也影响流构造。V2诊断修正后覆盖率从约48%-50%升至99%以上。K训练使用的是旧V1流程，因此其退化不能单独否定轨迹速度或光流方向。[13]')
heading('L轮隔离了共享应力的作用')
p('L0/L1都修正流与相机辅助流程、都由完整中心R(q)X+b导出速度，只改变stress来源。L1验证PSNR仅+0.0198 dB、速度推进EPE仅下降0.12%，没有达到预设10%运动改善门槛。新增共享头确实更新，遗留物理网按协议冻结，不能解释为优化器漏接。[14]')
box('v由同一轨迹定义时，内部v=dx/dt误差为零是结构定义；真实速度、应力散度与图像目标能否同时正确，仍需要独立观测和数值稳定性证据。')
p('各行之间不是统一架构排名：K→L同时修辅助流程；L→M改变反向梯度策略；M→N同时改变离散时间和空间求导。M/N详见下一页。[12-16]','small')

start('07 / 本周 · 物理优化','图像恢复了，物理一致性仍不足','动量实验保留真实前向残差；没有用小反向梯度冒充小物理误差。')
heading('M1：动量梯度保护')
p('把共享形变参数的加权动量梯度限制在同批RGB梯度的10%，私有stress头与原应变率系数梯度保留。相对L1，best测试PSNR 33.9658→39.7018 dB，速度推进EPE 0.8363→0.7611 px（下降8.99%），未达到预先10%门槛。[15]')
p('共同64点探针的帧83动量RMS约4711.6，原应变率修正散度约4734.3，直接stress散度仅0.075。该诊断支持率修正/空间求导与梯度竞争仍是问题；不证明“缺外力”是根因。首个动量事件前已有曲线差异，全部图像恢复不能严格归为单项因果。[15]')
heading('N1：一致离散区间与二次局部求导')
table(['M1→N1','M1','N1','变化'],[
 ('best测试PSNR / dB','39.7018','39.9095','+0.2077'),
 ('验证轨迹EPE / px','0.7739','0.7344','下降5.11%'),
 ('验证速度推进EPE / px','0.7611','0.7318','下降3.85%'),
 ('共同点动量Huber中位数，原算子','119.157','115.870','下降2.76%'),
 ('共同点动量Huber中位数，新算子','114.981','106.689','下降7.21%'),
 ],[1.9,.7,.7,.85])
p('帧83异常缓解，但帧2两种算子都恶化；跨时刻动量下降未达到25%门槛。结果对时间差分步长和邻域仍敏感。保留为数值兼容研究候选，不宣布物理成功。[16]')
heading('保留原C1结构，只加光流的直接消融')
table(['C1旧best→新光流O1','关闭','开启'],[
 ('best测试PSNR / dB','40.3344','39.9006'),
 ('验证轨迹EPE / px','0.7423','0.7486'),
 ('验证速度推进EPE / px','1.2263','1.0248'),
 ],[1.85,1,1])
p('速度推进改善16.43%，但验证PSNR损失0.3705 dB超过0.2 dB护栏，综合收益为false。光流改善主要落在独立速度分支，未传递为更好形变轨迹。此处O1是“光流O1”，与上周“物体分组O1”不同。[17]','small')

start('08 / 本周 · 点与物体','关联链建立了，跨遮挡身份仍未知','建议的记录单位：检查点SHA → 局部身份域 → 图像物体 → 多个点库 → 固定行 → 每时刻支持。')
box('先验证图像物体的连续对应，再用完整可见足迹和中心运动绑定Gaussian；历史归属与当前可见支持必须分别保存。')
table(['工作 / 状态','新证据','边界'],[
 ('P1形态候选\n16k停止','991标记点；早期贡献存在，中晚期失去支持','形态loss未启用；不能说该loss训练后无效'),
 ('全时段支持审计\n只读完成','早期C1点库贡献69.96%@2、12.95%@74、0%@86/200','固定早期局部patch不是完整物体；零支持不等于真实身份错误'),
 ('运动历史原型\n无神经训练','人工隐藏试验预测9.5898→2.4890 px，下降74.05%','隐藏穿过/反弹可有相同可见观测；保留歧义'),
 ('局部表面对应\n冻结C1完成','16库、3232不同行；71/120帧有部分接受，平均贡献覆盖17.76%','是严格条件下接受覆盖；非身份准确率/模型正确率'),
 ('清晰片段与尺度复核\n只读完成','原2-14原生支持66.85%；晚期188-200为37.43%','400直接渲染与1600渲染缩图不同；低分辨率会改变足迹/门槛'),
 ('物体-多点库注册\n只读完成','早2-27积累2065历史归属行；两库并集后续贡献62.84%→65.23%','局部物体归属可积累；不跨自然遮挡填真实ID'),
 ],[1,1.65,1.8])
p('关联规则同时检查源/目标中心、完整alpha×T足迹、片段纯度和原始双向流。旧rasterizer元数据每像素只记录前20个贡献者；新的完整分组合成审计避免把截断归一化比例当全部贡献。[18-23]')
p('本页除P1的前13k新增训练步骤外，其余主要是数据/推理/对应诊断，没有Gaussian优化；局部对应的实现与核验完成，不代表新的PSNR提升。分离轮廓、稳定分组号、Gaussian行号重叠都不足以证明真实材料点身份。[18-23]','small')

start('09 / 本周 · 关联链训练','短时关联更干净，长期支撑未恢复','11/2完成：同一C1 50k完整状态，T0与T1各追加2000次更新至52k；86,143点、2065锚点。')
table(['最终验证集 / 30张','T0普通续训','T1关联续训'],[
 ('PSNR / dB','39.5423','39.5355'),
 ('标准VGG LPIPS','0.014704','0.014724'),
 ('早期库贡献占鸭子alpha / 2-27','75.14%','65.22%'),
 ('库贡献落到鸭区外的比例 / 2-27','1.4001%','0.0963%'),
 ('中心与足迹同时支持的平均点数','1757.7','1859.8'),
 ('晚期146-200点库贡献','0%','0%'),
 ],[1.9,1,1])
p('局部纯度改善，但总贡献缩小。原帧82及以后的全部已评估训练时刻，早期库对可见鸭子均失去支撑；晚期29个训练帧对的时序关联项全部为零。当前可见光流没有提供跨完全遮挡的独立身份桥。[24]')
heading('11/3新增固定视角完整度metric：只读评估')
table(['细薄支撑P95 / %','原轨迹','固定2','固定86','固定188'],[
 ('C1','1.097','2.780','4.988','2.163'),
 ('T1 关联','1.131','2.765','4.946','2.149'),
 ('O1 光流','1.819','3.098','4.855','2.454'),
 ],[1.3,.8,.8,.8,.8])
p('在400×225的alpha>0.5支撑上，用半径2像素开运算测量细薄面积。另记录小孤岛、封闭孔洞、边界复杂度；有真实观测时才计算缺失率、核心alpha不足、背景外溢和IoU。[26]')
box('metric用于追踪，loss才影响训练。无GT的新视角代理不能直接当正确形状目标；最少孔洞/最少连通块会鼓励填平间隙、粘连不同物体。')
p('2400张既有alpha评估完成，新增12项语义测试加10项现有测试通过；没有训练器变更或新优化步骤。此页的P95是全场景alpha统计，不是“物体完整率”。[26]','small')

start('10 / 本周 · 关键图','固定相机扫描揭示时间-视角耦合','固定相机2/86/188，每台扫描200时刻；另保留原相机轨迹。C1/T1/O1共2400次原生渲染。[25]')
fig('output/fixed_camera_temporal/key_comparison.png',405,'图1  仓库原有关键对照图。四行依次为原轨迹GT、原轨迹C1、固定相机86的C1、固定相机alpha；各列时间相同。首行不是固定相机GT。图中英文/数值为原始实验标注。')
box('与实际观测重合的相机/时间组合较完整；离开该组合，RGB与alpha都出现拉丝、碎片、缺口。关联续训与光流训练没有消除这一模式。')
p('每个时刻只有一个相机，而非同一时刻的同步多视图。训练约束的是Render(G(t_k), c_k)，固定视角审计检查Render(G(t_k), c_fixed)。更多模型生成视角只能增加自洽检查，不能凭空提供真实运动。[25]')
p('主要伪影也存在于1600×900原生图，不是缩图或视频压缩产生。三固定相机矩阵指纹不变；16视频各200帧完整解码。人工复核覆盖预设关键帧与接触表，不冒称逐像素看完2400张图。没有独立固定新视角GT，因此未计算伪PSNR或真实3D表面误差。[25]','small')

start('11 / 本周 · 第一版','把现有观测迁移为几何约束','11/3：真实观测alpha覆盖 + 部分静态位置/协方差 + 三角化锚点 + 短轨迹投影 + 同组件局部形态。')
heading('首版：在C1 50k状态上匹配续训')
table(['相同50k起点→52k','U0关闭','U1开启'],[
 ('实际新增更新','2000','2000'),
 ('测试PSNR / dB','40.2854','40.2911'),
 ('标准VGG LPIPS','0.013926','0.013904'),
 ('短轨迹EPE / px@400','0.5431','0.3590'),
 ],[1.8,1,1])
p('16,092个条件静态行、42锚点、33,348条局部投影观测；相机采样2000条完全匹配，学习率不重置。EPE下降约33.9%，测试PSNR仅+0.0057 dB，两组都低于原始C1；固定视角仍碎裂，早期点库晚期贡献仍为零。[27]')
heading('第二版：检验“旧参数/训练太晚”的解释')
p('全新seed20261103，保存空优化器的零步状态，不加载历史学习张量。共同预热15k通过训练数据绑定门槛后分叉；S0关闭/S1开启，各训练至总50k，后35k相机采样完全一致，沿用完整学习率计划。[28]')
table(['第二版原生PNG评估','S0从零对照','S1从零约束'],[
 ('验证PSNR / dB','40.4663','40.4331'),
 ('测试PSNR / dB','40.8475','40.8205'),
 ('测试SSIM','0.994953','0.994927'),
 ('标准VGG LPIPS','0.012948','0.012984'),
 ('黄色外观区域PSNR / dB','28.1162','28.1126'),
 ],[1.8,1,1])
box('从零训练使比较更能检验早期接入的作用，但没有带来本轮图像收益。两组高于历史C1的差异包含新种子与运行差异，不能归功于新增约束。')
p('注意命名：本页“从零S1”是旧实验臂；后文“Stage S1”是独立静态重建阶段。两者不是同一个模型。[27,28,36]','small')

start('12 / 本周 · 第二版','局部损失大幅下降，画面仍碎裂','两组均88,916点；14624静态行、37锚点、31897短轨迹观测。[28]')
table(['同样本终态诊断','S0','S1','变化'],[
 ('短轨迹EPE / px@400','1.8039','0.2800','下降84.5%'),
 ('局部形态违约','0.353873','0.003381','下降99.0%'),
 ('静态锚点距离 / 场景单位','0.47937','0.16306','下降66.0%'),
 ('静态位移分量RMS','0.21431','0.19042','下降11.1%'),
 ('静态协方差损失','0.15304','0.14240','仍高于15k的0.09749'),
 ],[1.7,.8,.8,1.25])
fig('output/scratch_integrity/native_inspection_fixed86.png',320,'图2  固定相机86：左S0、右S1，预设早/中/晚原生图对照。与观测重合的中间时刻较完整；早/晚时刻仍拉丝、分层和破碎。为版面等比缩放，未补洞。')
p('早期点库贡献：早期55.80%→58.31%，中段14.29%→14.89%；晚期S0约1.94e-7（原始比例）、S1为0，均可忽略。局部约束确实生效，但没有恢复早期表面对晚期可见鸭子的持续支撑。[28]')
p('35项单测、真实GPU/Adam与恢复核验、2400次原生渲染、16视频各200帧解码完成。此结果削弱“只是旧参数或末期低学习率”的解释，但不否定所有从零或其他监督设计。没有跨遮挡身份恢复证据。[28]','small')

start('13 / 本周 · 静态表示','硬冻结子集会被优化绕开','R0完成并拒绝晋级；R0a定位初始参考问题；随后改为独立Stage S1静态重建。[29-31]')
table(['R0同15k起点→50k','H0动态对照','H1硬静态池'],[
 ('测试PSNR / dB','40.8464','40.7976'),
 ('标准VGG LPIPS','0.012932','0.013547'),
 ('验证静态蓝色可见贡献','51.60%','12.12%'),
 ('选中行opacity中位数','0.999996','0.003908'),
 ('静态图像轨迹重投影 / px@400','1.8430','7.5642'),
 ],[1.9,1,1])
p('硬静态行的时变中心/协方差误差按结构为零，但68.18%选中行opacity<0.01，动态/未知行承担拟合；固定视角仍碎裂。图像护栏通过，几何目标未通过，H1不晋级。[29]')
heading('R0a：问题不全由终态漂移造成')
p('同一固定映射在共同15k起点已有7.7975 px重投影误差，终态7.5642 px。14,624行来自120个参考帧，逐行冻结旧warp并未保证统一一致表面。37条条件轨迹不能代表全部静态几何。[30]')
heading('Stage S1：独立静态重建，取得条件进展')
table(['S1独立训练12k','验证30张','测试50张'],[
 ('静态候选区域PSNR / dB','35.6438','35.7912'),
 ('区域平均alpha','0.997189','0.997191'),
 ('已知背景平均alpha','0.002642','0.002614'),
 ],[1.8,1,1])
p('从训练多时刻轮廓支持初始化4126点；无动态参数。opacity中位数0.6584，低于0.01仅0.291%，未重演R0大规模透明化；可作为S2条件静态初始化。[31]')
box('这是局部可信度的进展。新分支采用metadata CV相机，历史PIDG相机不变；S1与R0架构/相机/目标不同，不能作一项改动的匹配消融，也不能宣布获得精确接触表面。')

start('14 / 本周 · 分阶段路线','Stage S2-S6：正式终态已复核','此处为重建Stage编号。以下动态PSNR均为各阶段原生1600×900条件动态区域验证值。[32-38]')
table(['阶段 / 实际更新','动态PSNR','动态alpha','静态PSNR','验收状态'],[
 ('S2 RGB首版 / 16k','9.3514','0.49449','33.8937','失败：动态与静态保持均未过'),
 ('S2 第一次修正 / 16k','25.0864','0.98499','31.3350','失败：静态退步；组scale退化'),
 ('S2 第二次修正 / 16k','24.2534','0.98265','35.5301','数值过门槛；几何未解决'),
 ('S3 物理场 / +6k','25.3923','0.98551','35.9417','可条件接续；仍有碎片/组泄漏'),
 ('S4 稀疏轨迹 / +4k','25.3402','0.98619','36.1994','轨迹误差反而恶化'),
 ('S5 条件内部先验 / +4k','25.6037','0.98482','35.9831','先验未激活；无碰撞修正证据'),
 ('S6 联合微调 / +4k','28.3270','0.98598','36.1699','观测/轨迹改善；新视角仍碎裂'),
 ],[1.3,.7,.7,.7,1.9])
p('S2使用四组固定canonical点和四个原PIDG形变网络，15,507动态点，S1静态几何保持冻结。正值有界scale和静态RGB归一化修复了部分退化；固定86/188仍可见扁片、拉丝，帧42组混合，晚期物理身份未知。[32]')
p('S3首个恢复短训因canonical邻域退化失败，记录保留；独立修正为当前世界同组邻域并显式恢复图缓存。随后完整6k与136个Adam状态/采样恢复核验通过，不能据此称应力真实或几何成功。[33]')
heading('S4/S5：必须同时报告负结果与目标是否激活')
p('S4绑定91条片段、455次观测、79个固定行，全部观测重投影0.53684→0.78502 px@400，恶化46.2%。S5降至0.66467 px，仍差于S3基线；均为训练图像对应，并非独立几何GT。[34,35]')
p('S5全程800次条件内部先验调用，活动违约和非零loss均为0。制造球/平面夹具通过300步算子测试，但真实场景没有实际碰撞修正；物体间非穿透、接触力、摩擦、恢复系数仍未验证/关闭。[35]')
box('S6正式4000步与原生评估完成：轨迹0.42711 px@400，较S3低20.44%；静态几何保持S1。观测改善，新视角仍碎裂；条件体先验800次均未激活。详见p31-34。[37,38]')

start('15 / 判断与下一步','目前能得出的结论','区分工程完成、目标优化、独立观测改善和物理真实性，避免把它们合成一个“成功”标签。')
table(['问题','目前答案','证据'],[
 ('原轨迹图像是否已较好？','是。C1/H3及从零对照取得较高RGB分数；仍有关键坏帧。','p3-5、p11'),
 ('S6是否优于历史H3？','共同test复评：全幅PSNR/SSIM/LPIPS全部50帧仍差；动态区域PSNR均值略升，27/50帧提高。','p35-46'),
 ('独立速度与形变是否一致？','原独立头明显不一致；轨迹导出能结构上统一读出，未使动量残差自动变好。','p5-7'),
 ('点能否归到物体？','清晰连续片段内，部分点可经图像轨迹和多点库建立局部物体归属。','p8-9'),
 ('遮挡后身份是否恢复？','尚未验证。完全隐藏交互缺独立身份线索；早期库晚期支持仍接近零。','p8-12'),
 ('固定视角是否完整？','当前检查的新组合仍碎裂；S6观测改善，固定86/188其他时刻仍有空洞和拉丝。','p10-14、p33-34'),
 ('物理/材料是否识别成功？','没有。共享E、内部stress/残差或零先验不提供材料/接触真值。','p5-7、p14'),
 ],[1.1,2.8,.6])
heading('后续优先级：补充可核验的几何与身份证据')
p('1. S6已完成：观测改善、455次既有轨迹误差下降，但固定新视角仍碎裂。接受为条件重建终态，不据阶段完成宣称完整几何或物理成功；本报告不启动额外训练。[37,38]')
p('2. 优先扩充可核验的跨间断物体对应和全时段表面支持；能取得合成器object/mesh ID、持久顶点或同步多视图时，先用其验证。缺证据时明确未知，避免最近点强配。')
p('3. 将静态参考、动态canonical/实际足迹、相机约定分别做可区分的短对照；固定视角代理只能辅助筛查，不能作为唯一正确几何目标。')
p('4. 重建与身份依据稳定后，再按M0-M10顺序推进F、本构一致性、共享材料参数和可辨识性；继续保留flags-off与单测/短训/恢复证据。')
p('前文保留历史实验口径；p35-47新增冻结模型的共同test逐帧复评，只读推理，未训练或修改PIDG。原实验预算和选择规则均保留，不将单次微小均值差解释为统计显著。','small')

ATLAS_FIGURE_COUNT=render_atlas(ATLAS,start,p,fig,box)
S6_FIGURE_COUNT=render_s6(S6_ASSETS,S6_DATA,start,p,fig,box,table,heading)
COMPARE_FIGURE_COUNT=render_comparison(COMPARE_ASSETS,COMPARE_DATA,start,p,fig,box,table,heading,compact_comparison_table)

for index,(lo,hi) in enumerate(((0,18),(18,38))):
    start(f'{PAGE+1:02d} / 来源索引','来源与复查入口' if index==0 else '来源与快照信息',
          '编号对应正文方括号；路径均相对于 F:/A_Edinburgh/PhysGauss。报告读取的是独立快照。')
    for i in range(lo,hi):
        title,rel=REFS[i]
        text=f'[{i+1:02d}] {title}\n{rel}'
        if i == 0:
            text += '\nexperiments/v3_iso_scratch/results.md\nexperiments/v3_isotropic/results.md'
        if i in (36,37): text+='\n（S6补充快照）'
        p(text,'source',5)
    if index==0:
        p('旧报告用于汇总上周实验；关键负结果以对应assessment和11/1审计复核。独立只读诊断未冒称完整训练，未运行项未进入成功/失败统计。','small')
    else:
        p(f'历史快照：{CUTOFF}；Git HEAD：{MANIFEST["git_head"][:12]}\nS6补充快照：{S6_CUTOFF}；正式终态SHA：{S6_MANIFEST["checkpoint_sha256"][:16]}\n历史来源见同名.sources.json/.sources.images.json；S6见.sources.s6.json，[36]仍为历史版本。共同test复评来源、PNG/权重/检查点SHA及逐帧数据见.test_comparison.json/.csv；它包含新增只读推理，无训练更新。','small')

QA.append({'page': PAGE,'bottom_content_y':round(Y,2)})
assert PAGE == TOTAL_PAGES, PAGE
c.save()
reader=PdfReader(str(PDF))
assert len(reader.pages)==TOTAL_PAGES
texts=[page.extract_text() for page in reader.pages]
assert all(len(t)>120 for t in texts)
for needle in ['40.8782','84.5%','0.78502','800','0.66467','68.18%','17.76%']:
    assert needle in '\n'.join(texts),needle
(OUT/(BASE+'.source.txt')).write_text('\n'.join(TRANSCRIPT),encoding='utf-8')
(OUT/(BASE+'.sources.json')).write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/(BASE+'.sources.images.json')).write_text(json.dumps(ATLAS_MANIFEST,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/(BASE+'.sources.s6.json')).write_text(json.dumps(S6_MANIFEST,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/(BASE+'.test_comparison.json')).write_text(json.dumps(COMPARE_DATA,ensure_ascii=False,indent=2),encoding='utf-8')
qa={'pages':len(reader.pages),'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
    'cutoff_utc':MANIFEST['cutoff_utc'],'layout':QA,'text_characters':[len(t) for t in texts],
    'atlas_pages':ATLAS_PAGE_COUNT,'atlas_figures':ATLAS_FIGURE_COUNT,
    's6_pages':S6_PAGE_COUNT,'s6_figures':S6_FIGURE_COUNT,
    's6_revision_utc':S6_MANIFEST['captured_utc'],
    'test_comparison_pages':COMPARE_PAGE_COUNT,'test_comparison_figures':COMPARE_FIGURE_COUNT,
    'visual_review':'pending rendered-page inspection'}
(OUT/(BASE+'.qa.json')).write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(PDF),'pages':len(reader.pages),'cutoff':CUTOFF,'qa':QA},ensure_ascii=False,indent=2))
