"""Build the English report from saved metrics, code receipts and GT/render PNGs.

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
TMP = ROOT / 'tmp/pdfs/pidg_report_en'
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
PDF = OUT / 'PIDG_Variants_Report_EN.pdf'
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

pdfmetrics.registerFont(TTFont('CN', 'C:/Windows/Fonts/arial.ttf', subfontIndex=0))
pdfmetrics.registerFont(TTFont('CN-Bold', 'C:/Windows/Fonts/arialbd.ttf', subfontIndex=0))
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
for name, size, leading, color in [('body', 10.2, 15.5, INK), ('small', 8.5, 13, GRAY),
                                  ('caption', 8.5, 13, GRAY), ('table', 8.2, 11.5, INK),
                                  ('thead', 8.2, 11.5, colors.white)]:
    styles[name] = ParagraphStyle(name, fontName='CN', fontSize=size, leading=leading,
                                 textColor=color, wordWrap='LTR', alignment=TA_LEFT)

c = canvas.Canvas(str(PDF), pagesize=(W, H), pageCompression=1)
c.setTitle('PIDG Constitutive Extensions: Variant Architectures and Results')
c.setAuthor('PhysGauss / PIDG Extension Project')
c.setSubject('Rubber Duck 50k experiments, GT/render comparisons and conditional identifiability')
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
        c.drawString(LEFT, 30, 'Rubber Duck | 50k | One scene, one seed | Results: 19 Oct 2026')
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
    box(0, 159, 106, 45, 'Canonical Gaussians X', 'Appearance / covariance')
    box(134, 159, 105, 45, 'Deformation D(X,t)', 'Rotation / translation')
    box(268, 159, 105, 45, 'Deformed x(t)', 'Original rendering state')
    box(400, 159, 106, 45, 'Rendered image', 'Image loss against GT')
    arrow(106,181,134,181);arrow(239,181,268,181);arrow(373,181,400,181)
    box(0, 77, 106, 45, 'Velocity / stress field', 'Retain predicted stress')
    box(134, 77, 105, 45, 'Local estimate of F', 'Neighbours in X and x(t)',True)
    box(268, 77, 105, 45, 'Neo-Hookean', 'Shared mu and kappa',True)
    box(400, 77, 106, 45, 'Stress consistency', 'Predicted vs. constitutive',True)
    arrow(186,159,186,122);arrow(320,159,320,134);arrow(320,134,186,134)
    arrow(239,99,268,99);arrow(373,99,400,99)
    arrow(53,122,53,139);arrow(53,139,453,139);arrow(453,139,453,122)
    box(134, 0, 239, 46, 'Structured external force f(X,t)', '36 shared coefficients; enters the original momentum residual',True)
    box(0, 0, 106, 46, 'Momentum residual', 'r_legacy - f')
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

# 1. Executive summary. English counterpart of the Chinese report.
p=Page(1,'REPORT / EXECUTIVE SUMMARY','PIDG Constitutive Extensions',
       'Architectures and results of five variants | English edition: 20 October 2026')
p.text('Based on the Rubber Duck 50k experiments, the reused original compatibility baseline, and the saved GT/render comparisons. Experimental snapshot: 19 October 2026. No additional training was performed for this report.',gap=18)
p.callout('<b>Main finding:</b> A2b, with learnable shared material parameters, achieves the highest PSNR: <b>28.409 dB</b>, or <b>+1.152 dB</b> over the reused baseline. The baseline retains the lowest LPIPS. Adding external force and staged optimization does not yield consistent improvements.')
p.heading('What changes in the architecture?')
p.text('The original PIDG Gaussian representation, deformation field, velocity/stress field and renderer are retained. An independent branch estimates a local deformation gradient F and converts it into Neo-Hookean stress. A consistency loss connects this stress to the original prediction. Later variants enable shared material learning, a low-dimensional force field and staged optimization.')
p.table([['Item','Coverage in this report'],['Long training','A1, A2a, A2b, A4_single and A6_single; 50,000 iterations each'],['Baseline','A0 is reused from a completed compatibility reproduction'],['Rendering','The same 50 test views; GT aligned with each of the five models'],['Materials','One shared material in Rubber Duck; known two-material support tested only on manufactured data']], [88,CW-88])
p.heading('Scope of the conclusions')
p.text('The results establish executable modules and descriptive image-quality differences in a single run. They do not establish accurate material or force recovery, or stable gains across scenes and seeds. A1 adds no new training loss, so its gain over the historical baseline cannot be attributed directly to the F estimator.')
p.text('Reading guide: protocol 02; architecture 03-05; quantitative results 06; GT comparisons 07-09; physical diagnostics 10; identifiability 11; controlled validation 12; conclusions and sources 13.','small')
p.finish()

# 2. Protocol.
p=Page(2,'EXPERIMENTAL PROTOCOL','Settings and comparison protocol',
       'Training budget, evaluation data and snapshot selection define how the results should be interpreted.')
p.table([['Item','Setting'],['Scene / split','Rubber Duck: 150 training views and 50 test views'],['Training budget','50,000 iterations per new variant; width 1,600; seed 0; original use_flow=False'],['Image metrics','Project evaluation scripts: PSNR, SSIM, VGG-LPIPS and MS-SSIM'],['Image snapshot','Saved best-snapshot test averages; not uniformly the final 50k state'],['Physics snapshot','Final 50k full checkpoints for physical logs and conditional Jacobians'],['Baseline provenance','Completed A0 reused; full historical source hashes were not recorded'],['Scene scope','No additional D-NeRF/HyperNeRF runs; no solid NH loss on smoke or cloth']], [100,CW-100])
p.heading('Best snapshots occur at different iterations')
p.table([['Variant','Completed iterations','Snapshot used for metrics / rendering']]+
        [[n,'50,000',f"{best[n]:,}"] for n in models],[106,126,CW-232])
p.text('The historical protocol selects the best snapshot by test PSNR. Because test data influence selection, these values are not unbiased estimates on an independently held-out evaluation set. Using the same views still permits a descriptive comparison within this protocol.','small')
p.text('A0 is a local compatibility reproduction of the released code. Windows/release compatibility fixes, incomplete historical source provenance and CUDA nondeterminism prevent attributing every difference to the new physics constraints. Sources: [1][2][3].','small')
p.finish()

# 3. Architecture.
p=Page(3,'METHOD / ARCHITECTURE','Retain PIDG; add a physics branch',
       'All new modules are configurable. The original predicted stress is retained.')
p.check(242);c.saveState();c.translate(0,p.y-220);diagram();c.restoreState();p.y-=239
p.text('Figure 1. Blue-grey boxes retain the original path; green boxes are extensions. Original regularizers are omitted from the diagram. The velocity, stress and momentum constraints remain in place.','caption')
p.text('Retained backbone: Grid4D encodes xyz and xyt/yzt/xzt hash features. Spatial gating and an MLP output centre transforms, rotation and scale offsets. The velocity/stress network uses six-plane features, Fourier time encoding and a shared MLP, with 3 velocity and 6 symmetric-stress outputs.','small')
p.table([['Variant','F / constitutive branch','Material parameters','External force','Optimization'],
         ['A1','F diagnostics only','No new learning','Off','Original'],
         ['A2a','F + NH consistency','Fixed mu=1, kappa=10','Off','Joint'],
         ['A2b','F + NH consistency','Learnable shared mu, kappa','Off','Joint'],
         ['A4_single','As A2b','One shared material','Low rank; 36 parameters','Joint'],
         ['A6_single','As A4_single','One shared material','As A4_single','Stages A-B-C-D']],
        [71,112,111,103,CW-397])
p.text('A1 records F without adding a loss or learnable material/force variables. A2b opens two material scalars; A4_single adds 36 scene-shared force coefficients.','small')
p.text('A3 requires known material/interface labels that are unavailable for Rubber Duck. A5 is post-fit identifiability analysis and reuses A4 training, leaving five new models in the rendered comparison. Sources: [3][4].','small')
p.finish()

# 4. F estimation and constitutive consistency.
p=Page(4,'METHOD / DEFORMATION AND STRESS','From deformation to stress consistency',
       'F describes local material deformation and links geometry to shared material parameters.')
p.heading('1. Differentiable local deformation gradient')
p.text('Neighbourhood differences are formed from canonical centres X and the actual rendered centres x(t). The deformation network supplies a quaternion rotation and translation: x(t)=R(q)X+b. A distance-weighted local fit uses 16 canonical neighbours and an identity ridge prior.')
p.image(equation(r'$A=\sum_j w_j\Delta X_j\Delta X_j^T,\quad B=\sum_j w_j\Delta x_j\Delta X_j^T$', 'eq_F1'),38)
p.image(equation(r'$F=(B+\alpha I)(A+\alpha I)^{-1},\quad J=\det(F)$', 'eq_F2'),38)
p.text('The ridge alpha scales with the local neighbourhood. Queries are masked for fewer than 6 usable neighbours, excessive unregularized condition number, nonfinite values or J at or below 10<super>-6</super>. Confidence is detached. The canonical graph is refreshed periodically or when point counts change.')
p.heading('2. Compressible Neo-Hookean solid stress')
p.image(equation(r'$\sigma_{const}=\mu J^{-5/3}\operatorname{dev}(FF^T)+\kappa\frac{\ln J}{J}I$', 'eq_NH'),43)
p.text('Here mu is the shear parameter and kappa the bulk parameter. Learnable values use softplus(raw)+10<super>-6</super> to remain positive. One pair is shared over the scene, initialized at (1,10), with no free stiffness per Gaussian. Module tests cover approximately zero elastic stress under rigid rotation.')
p.heading('3. Sparsely evaluated consistency loss')
p.text('The retained predicted stress includes the original symmetric stress head and velocity-gradient correction. Elementwise Huber distance to the constitutive stress is averaged over matrix entries, weighted by the validity mask and detached confidence, then averaged over all queries. Weight: 10<super>-4</super>; at most 256 queries every 100 iterations after warmup.')
p.callout('Gradients reach F, the original stress branch and enabled material parameters. Invalid samples contribute zero and reduce the effective constraint coverage. A smaller loss alone does not establish more accurate material recovery. Source: [4].')
p.finish()

# 5. Force and staging.
p=Page(5,'METHOD / FORCE AND OPTIMIZATION','Structured force and staged training',
       'A4_single adds force to A2b. A6_single keeps this structure and changes the update schedule.')
p.image(equation(r'$f(X,t)=\sum_{r=1}^{3}\phi_r(X)\,a_r(t),\qquad \phi=[1,X_x,X_y]$', 'eq_force'),40)
p.text('Each a_r(t) is a 3D vector interpolated linearly between four control times. This gives 4 x 3 x 3 = 36 shared coefficients, initialized to zero. The spatial basis contains only a constant and linear X/Y terms. No free force is introduced per Gaussian.')
p.text('Force enters as body-force density through r_legacy - f. Magnitude regularization has weight 10<super>-4</super>, temporal coefficient-difference regularization 10<super>-3</super>, and sparsity weight 0. A4 uses material learning rate 10<super>-5</super> and force learning rate 10<super>-4</super>.')
p.heading('Actual A6 update schedule')
p.table([['Stage / iterations','Original model','Material / force','New constraints'],
         ['A / 1-4,999','Original learning rates','Frozen / force off','No new loss'],
         ['B / 5,000-9,999','Original learning rates','Fixed material / force off','Constitutive consistency'],
         ['C / 10,000-19,999','Frozen; densification off','Update mu, kappa and force','Constitutive + force residual'],
         ['D / 20,000-50,000','Learning rates x 0.1','Both learning rates x 0.1','Joint refinement']],
        [116,124,140,CW-380])
p.text('Stage-C entry/exit hashes confirm unchanged neural weights. Stage D starts after the original 15k densification cutoff, so it cannot recover densification skipped during C. Equal iteration counts therefore do not imply equal neural update budgets.','small')
p.callout('<b>Physical interpretation:</b> the retained velocity/momentum path queries fid/train_count, whereas rendered F, constitutive stress and new force use fid. These conventions differ. The residual is not claimed to represent force balance in calibrated physical units. Sources: [4][5].')
p.finish()

# 6. Quantitative comparison.
p=Page(6,'RESULTS / IMAGE QUALITY','Highest PSNR: A2b, with metric trade-offs',
       'Higher PSNR/SSIM/MS-SSIM and lower LPIPS are better. Values use saved best snapshots.')
data=[['Method','PSNR','Delta PSNR','SSIM','LPIPS','MS-SSIM']]
for name,m in zip(all_names,all_metrics):
    data.append([name,f"{m['PSNR']:.3f}",f"{m['PSNR']-baseline['PSNR']:+.3f}",f"{m['SSIM']:.6f}",f"{m['LPIPS']:.6f}",f"{m['MS-SSIM']:.6f}"])
p.table(data,[90,69,66,92,93,CW-410],highlight=4)
p.image(chart,190,'Figure 2. Dashed lines show A0. Truncated vertical axes expose small differences. A4_s and A6_s denote the single-material variants. No repeated runs are available, so no error bars are shown.')
p.text('<b>Material ablation:</b> A2b exceeds fixed-material A2a by 1.281 dB and A0 by 1.152 dB. Its LPIPS is 0.030052 versus A0 at 0.029624, so the improvement does not hold across all image metrics.')
p.text('<b>Added complexity:</b> A4_single is 0.080 dB below A2b; A6_single is 2.002 dB below A4_single. Staging is weaker for reconstruction in this run, but one experiment cannot isolate which component or schedule choice caused the difference.')
p.text('<b>A1 control:</b> A1 adds no training loss yet exceeds the old baseline by 0.893 dB. Same-version flags-off A0 runs, multiple seeds and consistent snapshot selection are needed before making causal claims about the new constraints. Sources: [1][2].','small')
p.finish()

# 7-9. Same three views and same GT-only crop as the Chinese edition.
comments={
    0:('Start of sequence','The initial view checks the spatial layout of duck bodies and blue supports. All variants preserve the main structure, while silhouettes, thin support edges and small parts still differ from GT. A single frame cannot substitute for test-set averages.'),
    25:('Middle of sequence','The middle view checks duck silhouettes and support edges during deformation. The overall layout is close to GT, with remaining errors in sharpness and local colour. Continuous videos are needed to assess jitter or temporal consistency; static panels do not quantify either.'),
    49:('End of sequence','The final view shows normalized time 1.0. The principal objects remain reconstructed, with persistent local detail errors. This is a reconstruction example under the test-selected snapshot protocol, not evidence of independent future prediction or recovered physical dynamics.')}
for number,idx in enumerate(frame_ids,7):
    stage,comment=comments[idx];view=manifest['views'][idx]
    p=Page(number,'RESULTS / GT AND FIVE MODELS',f'GT comparison: {stage}',
           f"Test index {idx:02d} / 49 | Source image {view['image_name']} | Normalized time t={view['time']:.3f}")
    p.text('Start, middle and end views were selected in advance. Every panel uses the same crop, determined only from the union of foreground in these three GT images. Scaling is identical across models; no colour enhancement is applied.','small')
    im=Image.open(visuals[idx]);height=min(462,CW*im.height/im.width)
    p.image(visuals[idx],height,f'Figure {number-4}. GT and all five variants with a fixed crop. Panel labels give each best-snapshot iteration. Original full-resolution views are 1600 x 900.')
    p.text(comment)
    p.text('A0 was not rerendered in this five-model gallery and appears only in the quantitative comparison. The complete gallery contains 50 frames, paired GT/render PNGs and 10 fps videos per model. Rerendered pixels match the previous evaluation renders exactly. Source: [6].','small')
    p.finish()

# 10. Final checkpoint diagnostics.
p=Page(10,'RESULTS / FINAL CHECKPOINT','Image gains do not establish material recovery',
       'This page uses final 50k logs; page 6 uses best-snapshot reconstruction metrics.')
data=[['Variant','mu','kappa','Raw constitutive loss','Invalid F']]
for n in models:
    ph=by_name[n]['physics']
    data.append([n,f"{ph['mu']:.6f}" if 'mu'in ph else 'N/A',f"{ph['kappa']:.6f}" if 'kappa'in ph else 'N/A',f"{ph['loss_constitutive_raw']:.6f}" if 'loss_constitutive_raw'in ph else 'Not enabled',f"{100*ph['invalid_F_fraction']:.4f}%"])
p.table(data,[100,88,95,125,CW-408])
p.text('Learnable material values remain close to initialization at mu=1 and kappa=10. For A2b, the changes are approximately 0.050% and 0.017%, respectively. Without material ground truth for Rubber Duck, these values cannot establish true parameter accuracy.')
p.heading('Logged stress and force magnitudes')
data=[['Final 50k quantity','A4_single','A6_single']]
for label,key in [('Mean constitutive stress norm','mean_sigma_const_norm'),('Mean predicted stress norm','mean_sigma_pred_norm'),('Mean stress-difference norm','mean_sigma_difference_norm'),('External-force term norm','external_force_norm'),('Total residual term norm','total_residual_norm')]:
    data.append([label]+[f"{by_name[n]['physics'][key]:.6f}" for n in ['A4_single','A6_single']])
p.table(data,[239,134,CW-373])
p.text('A6 has smaller constitutive loss and logged residual but worse image metrics. Optimizing these proxies does not guarantee better reconstruction. The stress branches are not fully aligned either, and the logs do not directly measure physical accuracy.')
p.text('Finite, nonzero gradients to F and predicted stress confirm connected gradient paths. A2b logs approximately 3.54 x 10<super>-5</super> and 8.86 x 10<super>-7</super> at 50k. Validity is measured on 256 sampled queries: 0% invalid does not certify an inversion-free full field. Sources: [1][4].','small')
p.finish()

# 11. Identifiability.
p=Page(11,'RESULTS / IDENTIFIABILITY','Conditional rank is not physical recovery',
       'Neural geometry, velocity and stress are frozen for the local physical-parameter Jacobian.')
diag4=by_name['A4_single']['identifiability']['conditional_physical_residual']
diag6=by_name['A6_single']['identifiability']['conditional_physical_residual']
p.table([['Final checkpoint','Rank / parameters','Condition number','Minimum singular value'],
         ['A4_single','38 / 38',f"{diag4['condition_number']:,.1f}",f"{diag4['min_singular_value']:.4f}"],
         ['A6_single','38 / 38',f"{diag6['condition_number']:,.1f}",f"{diag6['min_singular_value']:.4f}"]],
        [118,145,126,CW-389])
p.text('The 38 parameters are log mu, log kappa and 36 force coefficients. Results depend on the frozen neural fields, residual units, parameterization and numerical scales. Neural-field uncertainty is not marginalized, and the covariance proxy is not a noise-calibrated confidence interval.')
p.callout('The diagnostic concatenates stress-consistency residuals and the original momentum residual minus force. With neural fields frozen, material parameters affect only the first block and force only the second. The reported zero material/force cross-correlation is therefore structural to this conditional construction; it does not establish absence of confounding in the original inverse problem.')
p.heading('A separate manufactured ambiguity counterexample')
p.text('Construct mu*u + kappa*v - f*u = y, with u=sin(t), v=cos(t) and true (mu,kappa,f)=(2,3,1). When mu and f are both unknown, observations constrain only their difference. This is an analytic balance test, not inversion from official RGB images.')
p.table([['Setting','Data rank / parameters','Objective rank','mu relative error','Force relative error'],
         ['Known force','2 / 2','2','0%','0%'],['Known stiffness','1 / 1','1','0%','0%'],
         ['Both unknown','2 / 3','2','75%','150%'],['Both unknown + biased prior','2 / 3','3','25%','50%']],
        [136,110,72,91,CW-409])
p.text('With both unknown, the condition number is infinite and the material/force correlation proxy is approximately 0.999978. A biased prior improves objective conditioning to 20.55, but observation-only rank remains deficient and parameters remain biased. Convergence, regularized full rank and recovery are distinct claims. Sources: [5][7].','small')
p.finish()

# 12. Controlled manufactured validation.
p=Page(12,'VALIDATION / TWO MATERIALS','Two-material tests and interface validation',
       'Engineering validation: 300 iterations at width 400. These are not 50k official-scene gains.')
p.text('Manufactured data contain two explicitly labelled regions and 128 material points. Multi-material versions use shared regional parameter banks without learned segmentation. True mu=[1,2] and kappa=[10,20]; multi-material initial values are 20% low. Images use a simplified disc renderer, and interface normals use a canonical centre-line proxy.')
data=[['Controlled configuration','PSNR','SSIM','LPIPS','Interface proxy loss']]
for row in results['smoke_rows']:
    n=row['configuration_id'].replace('_controlled','');ph=row.get('physics',{})
    interface=ph.get('loss_interface_raw',ph.get('interface_traction_raw'))
    if interface is None:
        candidates=[v for k,v in ph.items() if 'interface' in k and isinstance(v,(int,float))]
        interface=candidates[0] if candidates else None
    data.append([n,f"{row['metrics']['PSNR']:.3f}",f"{row['metrics']['SSIM']:.6f}",f"{row['metrics']['LPIPS']:.6f}",f"{interface:.6f}" if interface is not None else 'Not enabled / logged'])
p.table(data,[145,72,91,91,CW-399])
p.text('Image quality does not improve monotonically as modules are added. These runs validate executable configurations and logging, rather than material or force recovery. Real multi-material A3 and full A4/A6 with genuine interface labels still await known material_id annotations.')
p.heading('Module tests and parameter recovery are different evidence')
p.text('Saved milestone validation covers affine F recovery, near-zero NH stress under rigid rotation, constitutive backpropagation, region-shared parameters, finite force gradients, stage freezing and flags-off paths. In an earlier independent M9 controlled smoke test, the conditional Jacobian was full rank at 40/40, yet material errors remained about 20% and force error about 100%.')
p.text('Passing module tests must therefore be reported separately from successful physical identification. Rubber Duck has no available material/force ground truth, so no physical recovery error is supplied for that scene. Sources: [1][3][7][8].','small')
p.finish()

# 13. Conclusions and local references.
p=Page(13,'CONCLUSIONS / SOURCES','Conclusions and next validation steps',
       'Separate architectural effects, optimization budgets and historical run differences through fair controls.')
p.text('<b>Model choice:</b> A2b is the most promising candidate when PSNR is the priority in this run. A0 remains best for LPIPS. Added force in A4_single does not improve PSNR, and the current A6_single schedule is not the preferred reconstruction configuration.')
p.text('<b>Interpretation:</b> the constitutive branch is usable as a sparse geometry/stress consistency regularizer. Near-initial material values, differing time conventions and conditional diagnostic limitations do not support a claim of recovered true materials or forces. A1 also motivates controls for nondeterminism and historical baseline differences.')
p.text('<b>Recommended order:</b> (1) rerun same-version, same-environment flags-off A0 and A1 with independent validation-based selection and multiple seeds; (2) revisit A2a/A2b gains and loss frequency/weight; (3) match neural updates and densification budgets for A4/A6; (4) study consistent physical time/units in a separate experiment, then test material, force and joint identifiability against known truth or labels.')
p.heading('Local sources and reproduction entry points')
sources=[
    '[1] experiments/results.json and results.md: long runs, controlled comparisons and final physical-log summaries.',
    '[2] baseline_metrics.json; PIDG_REPO_MAP.md: compatibility baseline, code mapping and known differences.',
    '[3] experiments/configs/*.json; experiments/pidg_experiment_matrix.yaml: flags and scene scope.',
    '[4] pidg_ext/deformation_gradient.py, constitutive.py, losses.py and runtime.py: F, NH, loss and training integration. Backbone: variants/constitutive_pidg/code/scene_PIDG/network.py and motion_utils/pidg.py.',
    '[5] pidg_ext/forces.py, staging.py and identifiability.py; scripts/diagnose_constitutive.py: force, schedule and conditional diagnostics.',
    '[6] runs/pidg_constitutive/render_comparisons/rubber_duck_best_20261019_001152/manifest.json and delivery_validation.json: snapshots, views and pixel checks.',
    '[7] experiments/M9_validation.json; runs/pidg_constitutive/m10_long_A4_single/rubber_duck/identifiability.json and the corresponding A6 file: ambiguity and final diagnostics.',
    '[8] experiments/M1_validation.json through M10_validation.json; experiments/milestones.json: saved milestone validation and status.',
]
for line in sources:p.text(escape(line),'small',gap=6)
p.text('English build script: scripts/build_pidg_report_en.py. Tables read saved JSON and figures use validated PNGs. The full gallery remains beside [6]: index.html, per-model gt_render.mp4 and gt_and_five_models.mp4. This English edition preserves the original report\'s experimental scope and conclusions.','small',gap=0)
p.finish();c.save()

provenance_paths=[ROOT/'experiments/results.json',ROOT/'baseline_metrics.json',ROOT/'PIDG_REPO_MAP.md',
                  ROOT/'scripts/diagnose_constitutive.py',ROOT/'experiments/milestones.json',
                  ROOT/'runs/pidg_constitutive/m10_long_A6_single/rubber_duck/optimization_stages.jsonl',
                  RENDER/'manifest.json',RENDER/'delivery_validation.json',Path(__file__),
                  ROOT/'scripts/build_pidg_report.py',OUT/'PIDG_Variants_Report.pdf']
provenance_paths+=sorted((ROOT/'pidg_ext').glob('*.py'))+sorted((ROOT/'experiments/configs').glob('*.json'))
provenance_paths+=sorted((ROOT/'experiments').glob('M*_validation.json'))
provenance_paths += [ROOT/'variants/constitutive_pidg/code/scene_PIDG/network.py',
                    ROOT/'variants/constitutive_pidg/code/motion_utils/pidg.py']
provenance={'pdf':str(PDF),'language':'en','english_edition_date':'2026-10-20','experimental_snapshot_date':'2026-10-19',
            'pages':page_records,'fixed_gt_roi_xyxy':roi,'visual_indices':frame_ids,
            'best_snapshot_iterations':best,'source_sha256':{str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in provenance_paths},
            'limitations':['one scene, one seed','best selected by test PSNR','A0 historical source hashes unavailable',
                           'image best snapshot differs from final physics checkpoint','conditional Jacobian freezes neural fields',
                           'legacy CMR and new constitutive/force time conventions differ']}
(OUT/'PIDG_Variants_Report_EN.provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(PDF),'pages':len(page_records),'roi':roi,'content_bottom':[q['content_bottom'] for q in page_records]}))

