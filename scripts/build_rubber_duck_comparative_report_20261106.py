"""Evidence-based Chinese PDF; historical experiment inputs remain read-only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image as PILImage,ImageDraw,ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'output/pdf';TMP=ROOT/'tmp/pdfs/rubber_duck_comparison_20261106'
PDF=DEST/'RubberDuck_实验比较技术报告_20261106.pdf'
FONT=Path('C:/Windows/Fonts/msyh.ttc');BOLD=Path('C:/Windows/Fonts/msyhbd.ttc')
NAVY='#16354A';TEAL='#087E8B';AMBER='#C97818';INK='#263746';MUTED='#62717D';LIGHT='#EDF4F6'
W,H=A4;CW=W-84
sources={};figures={};story=[];page_specs=[]


def read(path):
    path=ROOT/path;sources[path.relative_to(ROOT).as_posix()]=digest(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_text(path):
    p=ROOT/path;sources[path]=digest(p);return p.read_text(encoding='utf-8-sig')


def register():
    DEST.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
    pdfmetrics.registerFont(TTFont('ZH',str(FONT),subfontIndex=0))
    pdfmetrics.registerFont(TTFont('ZHB',str(BOLD),subfontIndex=0))
    pdfmetrics.registerFontFamily('ZH',normal='ZH',bold='ZHB',italic='ZH',boldItalic='ZHB')
    font_manager.fontManager.addfont(str(FONT));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(FONT)).get_name(),
        'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#BCC9CE','axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})


STYLES={}
def styles():
    for key,size,leading,color,space in [('body',10.4,17,INK,7),('small',8.8,13,MUTED,5),('cap',8.2,12.2,MUTED,8),
        ('h1',21,29,NAVY,14),('h2',12.2,19,TEAL,9),('table',8.8,13.5,INK,0),('tablehead',8.8,13.5,'#FFFFFF',0),
        ('formula',9.2,16,NAVY,10),('kicker',8.5,13,TEAL,6)]:
        STYLES[key]=ParagraphStyle(key,fontName='ZHB' if key in ['h1','h2','tablehead'] else 'ZH',fontSize=size,
            leading=leading,textColor=colors.HexColor(color),spaceAfter=space,wordWrap='CJK',alignment=TA_LEFT)


def p(text,kind='body'):return Paragraph(text,STYLES[kind])
def add(text,kind='body'):story.append(p(text,kind))
def h(text):add(text,'h2')
def gap(n=8):story.append(Spacer(1,n))


def page(number,title,kicker):
    if story:story.append(PageBreak())
    page_specs.append((number,title));add(f'{number:02d} / {kicker}','kicker');add(title,'h1')


def table(headers,rows,widths=None):
    widths=widths or [CW/len(headers)]*len(headers)
    values=[[p(str(v),'tablehead') for v in headers]]+[[p(str(v),'table') for v in row] for row in rows]
    t=Table(values,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F1F6F7')]),
        ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#C9D6DA'))]))
    story.extend([t,Spacer(1,10)])


def note(title,text):
    t=Table([[p(title,'h2')],[p(text)]],colWidths=[CW-22]);t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),colors.HexColor(LIGHT)),('BOX',(0,0),(-1,-1),.5,colors.HexColor('#D5E3E7')),
        ('LEFTPADDING',(0,0),(-1,-1),11),('RIGHTPADDING',(0,0),(-1,-1),11),('TOPPADDING',(0,0),(-1,0),9),('BOTTOMPADDING',(0,-1),(-1,-1),9)]))
    story.extend([t,Spacer(1,11)])


def img(path,caption,maxh=310):
    im=PILImage.open(path);width=CW;height=width*im.height/im.width
    if height>maxh:height=maxh;width=height*im.width/im.height
    story.extend([Image(str(path),width=width,height=height,hAlign='CENTER'),Spacer(1,5),p(caption,'cap')])


def savefig(name,fig):
    path=TMP/f'{name}.png';fig.savefig(path,dpi=220,bbox_inches='tight');plt.close(fig);figures[name]=str(path);return path


def source_image(path):
    path=ROOT/path;sources[path.relative_to(ROOT).as_posix()]=digest(path);return PILImage.open(path).convert('RGB')


def panel(name,cells,columns,cell=(560,360)):
    cw,ch=cell;rows=(len(cells)+columns-1)//columns
    canvas=PILImage.new('RGB',(columns*cw,rows*(ch+38)),NAVY);draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(str(FONT),19)
    for i,(label,im) in enumerate(cells):
        x=(i%columns)*cw;y=(i//columns)*(ch+38)
        canvas.paste(im.resize((cw,ch),PILImage.Resampling.LANCZOS),(x,y+38));draw.text((x+10,y+8),label,font=font,fill='white')
    out=TMP/f'{name}.jpg';canvas.save(out,quality=96,subsampling=0);figures[name]=str(out);return out


def crop(path,cid):
    image=source_image(path)
    box=(290,160,1260,820) if cid==86 else (320,80,1200,810)
    assert image.size==(1600,900),path
    return image.crop(box)


def charts(ev,controller):
    # Each pair is independently matched; these are not repeated seeds.
    fig,ax=plt.subplots(figsize=(8.6,3.3))
    names=['L4 时序记忆','L5 表面支撑','L6 可撤销弹簧','L8-R 相对顺序']
    control=np.array([31.636682,31.223379,31.938599,ev['validation']['means']['rgb_control']['PSNR']])
    candidate=np.array([31.319321,31.859706,31.589028,ev['validation']['means']['ordinal_depth']['PSNR']])
    yy=np.arange(4)
    ax.hlines(yy,control,candidate,color='#CAD5D9',lw=4)
    ax.scatter(control,yy,s=55,c=TEAL,label='各轮匹配对照',zorder=3);ax.scatter(candidate,yy,s=55,c=AMBER,label='各轮候选',zorder=3)
    ax.axvline(31.994673,color=NAVY,ls='--',lw=1,label='J6父模型')
    ax.set(yticks=yy,yticklabels=names,xlabel='全30张验证图平均PSNR (dB)');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,frameon=False,fontsize=8)
    for i,(a,b) in enumerate(zip(control,candidate)):ax.text(min(a,b)-.025,i+.22,f'{b-a:+.3f} dB',fontsize=8,color=MUTED)
    compare=savefig('matched_PSNR_comparison',fig)
    fig,ax=plt.subplots(figsize=(8.5,2.7));x=np.arange(3);bw=.23
    for i,(name,vals,color) in enumerate([('L7-P (不同原分割)',[7,7,10],MUTED),('同扩展锚点逐帧',[12,9,5],TEAL),('扩展锚点联合',[17,19,3],AMBER)]):
        bs=ax.bar(x+(i-1)*bw,vals,bw,label=name,color=color)
        ax.bar_label(bs,padding=2,fontsize=8)
    ax.set(xticks=x,xticklabels=['早期 0-39','中期 40-79','晚期 80-119'],ylabel='标定通过视图数 / 各40');ax.set_ylim(0,23)
    ax.legend(ncol=3,loc='upper center',fontsize=8,frameon=False);ax.grid(axis='y',alpha=.12);calib=savefig('calibration_coverage',fig)
    fig,axes=plt.subplots(1,2,figsize=(8.6,2.8),sharey=True)
    for ax,(arm,label) in zip(axes,[('rgb_control','纯RGB对照'),('ordinal_depth','相对顺序候选')]):
        rows=[dict(completed_updates=0,active=19,suspended=11,revoked=0)]+controller['arms'][arm]['refreshes']
        xx=[r['completed_updates'] for r in rows];bottom=np.zeros(len(rows))
        for key,text,color in [('active','活跃',TEAL),('suspended','暂停','#E5BA69'),('revoked','撤销','#AAB6BC')]:
            vals=[r[key] for r in rows];ax.bar(xx,vals,70,bottom=bottom,label=text,color=color);bottom+=vals
        ax.set(title=label,xlabel='已完成训练更新',xticks=xx,ylim=(0,33));ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.1)
    axes[0].set_ylabel('固定30对关系资格');axes[1].legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.25),fontsize=8,frameon=False)
    ctrl=savefig('revocable_controller',fig)
    return compare,calib,ctrl


def visuals():
    base='output/staged_reconstruction/RubberDuck_relative_order/v1'
    current={}
    for cid in [86,188]:
        cells=[]
        for t in [2,100,200]:
            for model,label in [('J6_parent','J6'),('rgb_control','纯RGB'),('ordinal_depth','相对顺序')]:
                cells.append((f't{t:03d} / {label}',crop(f'{base}/c{cid:03d}_t{t:03d}_{model}_jointRGB_native.png',cid)))
        current[cid]=panel(f'fixed_{cid}_three_times',cells,3,(560,370))
    cells=[]
    for cid,t in [(188,100),(86,200)]:
        for kind,label in [('dynamicRGB','动态RGB'),('dynamicAlpha','动态alpha'),('dynamicD-alpha','动态D/alpha')]:
            cells.append((f'c{cid} t{t} / {label}',crop(f'{base}/c{cid:03d}_t{t:03d}_ordinal_depth_{kind}_native.png',cid)))
    decom=panel('dynamic_support_evidence',cells,3,(560,390))
    cells=[]
    methods=[('RubberDuck_relative_order','J6_parent','J6'),('RubberDuck_temporal_memory','temporal_memory','L4 时序'),
        ('RubberDuck_surface_support','surface_support','L5 结构'),('RubberDuck_revocable_springs','revocable_spring','L6 弹簧'),
        ('RubberDuck_relative_order','ordinal_depth','L8-R 顺序')]
    for kind,label in [('jointRGB','RGB'),('dynamicAlpha','动态alpha')]:
        for folder,model,name in methods:
            path=f'output/staged_reconstruction/{folder}/v1/c188_t100_{model}_{kind}_native.png'
            cells.append((f'{name} / {label}',crop(path,188)))
    historical=panel('historical_same_view',cells,5,(420,340))
    # Reformat already-rendered trajectory sheet without re-rendering any model.
    sheet=source_image(f'{base}/observed_train_index119.png');assert sheet.size==(800,1946)
    cells=[]
    for i,label in enumerate(['训练RGB观测','J6','纯RGB','相对顺序']):
        y=62+i*478;image=sheet.crop((0,y,800,y+450)).crop((120,35,670,420))
        cells.append((label,image))
    observed=panel('observed_trajectory_119',cells,4,(420,300))
    return current,decom,historical,observed


def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#D7E1E5'));canvas.setLineWidth(.5)
    canvas.line(42,32,W-42,32);canvas.setFont('ZH',7.3);canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(42,20,'PhysGauss / Rubber Duck 实验比较 | 2026-11-06')
    canvas.drawRightString(W-42,20,f'{doc.page:02d}')
    if doc.page>1:
        canvas.setFont('ZH',7);canvas.drawRightString(W-42,H-23,'观测、推断与几何真值分别报告')
    canvas.restoreState()


class ReportDoc(SimpleDocTemplate):
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name=='h1':
            key=f'section-{self.page}'
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(),key,level=0,closed=False)


def build():
    register();styles()
    b='experiments/staged_reconstruction/'
    assessments={name:source_text(b+folder+'/v1/assessment.md') for name,folder in [
        ('J6','RubberDuck_I10_joint'),('L4','RubberDuck_temporal_memory'),('L5','RubberDuck_surface_support'),
        ('L6','RubberDuck_revocable_springs'),('L7-P','RubberDuck_pseudo_depth'),('L7-J','RubberDuck_joint_scale'),
        ('L8','RubberDuck_ordinal_depth'),('L8-R','RubberDuck_relative_order')]}
    base=b+'RubberDuck_relative_order/'
    protocol=read(base+'protocol.json');ev=read(base+'v1/evaluation.json');ctrl=read(base+'v1/controller_audit.json')
    pairs=read(base+'v1/pairs_audit.json');final=read(base+'v1/final_review.json');audit=read(base+'v1/independent_audit.json')
    validation=read(base+'v1/final_validation.json');videos=read(base+'v1/videos.json');native=read(base+'v1/native_check.json')
    recovery=read(base+'v1/recovery.json');trainreceipt=read(base+'v1/training_receipt.json')
    train=read('data/derived/kinematic_resplit/training/transforms_train.json')
    val=read('data/derived/kinematic_resplit/transforms_val.json')
    assert len(train['frames'])==120 and len(val['frames'])==30
    assert len({f['time'] for f in train['frames']})==120
    assert len({str(f['transform_matrix']) for f in train['frames']})==120
    means=ev['validation']['means'];orders=ev['ordinal_diagnostic']['models'];usage=ctrl['arms']['ordinal_depth']
    assert usage['positive_auxiliary_steps']==20 and usage['used_pair_observations']==74
    assert all(q['passed'] for q in [pairs,ctrl,audit,validation,native,recovery])
    for path in ['variants/staged_reconstruction/i10_relative_order/core.py',
                 'variants/staged_reconstruction/i10_ordinal_depth/core.py',
                 'variants/staged_reconstruction/i10_relative_order/runtime.py',
                 base+'refresh_repair.md']:
        source_text(path)
    # Verify the prior numerical evidence rather than inventing an inter-run ranking.
    for name,value in [('L4','31.319321'),('L5','31.859706'),('L6','31.589028'),('L7-J','39张通过')]:assert value in assessments[name],(name,value)
    comparison,calibration,controller=charts(ev,ctrl)
    fixed,decomposition,historical,observed=visuals()

    page(1,'Rubber Duck\n重建实验比较报告'.replace('\n','<br/>'),'技术报告 / 训练、关系与完整性')
    add('从时序记忆、局部结构和弹簧，到伪深度标定与相对顺序训练','h2')
    add('报告日期：2026-11-06　　主要对象：J6、L4-L8-R　　版本：1.0','small')
    gap(4)
    note('结论先行','本轮已完成相对前后顺序候选与纯RGB对照各600步，但关系一致数量未增加，固定视角的碎裂和拖影没有一致改善。保留J6；新权重和全部负结果留存。')
    table(['当前模型/试验','全30张验证图PSNR','实测结论'],[
        ['J6父模型',f"{means['J6_parent']['PSNR']:.3f} dB",'当前保留的工作模型；本身仍有离轨迹缺陷'],
        ['纯RGB续训600步',f"{means['rgb_control']['PSNR']:.3f} dB",'匹配控制'],
        ['相对顺序600步',f"{means['ordinal_depth']['PSNR']:.3f} dB",'较对照-0.101 dB；视觉门槛未通过']], [155,114,CW-269])
    img(fixed[188],'封面示例：固定相机188，三时段、三模型的同区域原生图裁切。完整说明见第15页。',maxh=205)
    add('证据范围：本报告未新增训练、未修改任何冻结结果。相对顺序与D/alpha均非独立几何真值；父模型历史颜色条件来源仍披露。 [S1-S8]','small')

    page(2,'读完这份报告应得到什么','结论与导航')
    table(['问题','本次证据给出的回答'],[
        ['追踪/时序是否完全不起效？','不是。L4的已绑定记忆重投影误差下降，但只覆盖22行、偏早时段，未转化为全物体完整性。'],
        ['连接、厚度或弹簧是否有作用？','局部代理受到约束：L5连接间隙、L6平均应变改善；整体RGB、面积/覆盖尾部及离轨迹形状没有同步获益。'],
        ['伪深度能否直接做绝对监督？','L7-P与L7-J的尺度/动态覆盖未通过原门槛，未启动Gaussian训练。不能把有限尺度标定外推至全图。'],
        ['只学谁在前面是否就够了？','L8-R确实进行了训练，但固定fit一致数9/30、held一致数19/32均未变，尚不能可靠确定整鸭与环境的三维关系。'],
        ['下一步应该直接加训或加权吗？','现有证据不支持直接这样做。应先改善时空覆盖及关系到表面节点的可靠对应，再预声明一个固定候选。']], [151,CW-151])
    h('建议阅读顺序')
    add('第3-4页先明确数据和比较范围；第5-9页阅读时序、结构、弹簧及尺度审计；第10-13页查看本轮相对顺序训练的机制和数值；第14-17页集中看原图；第18-20页看解释、审计和来源。')
    note('三类证据不能混用','① 训练轨迹RGB：验证图像拟合；② 数据推断/父模型代理：验证假设的自洽性；③ 独立几何GT：验证真实三维。本报告相关Rubber Duck试验没有第三类固定时间多视角动态几何验证。')
    add('“未晋级”表示试验已完成但未替换当前工作模型，不表示模型没有更新或新权重未保存。 [S1-S8]','small')

    page(3,'数据、相机与证据定义','共同实验条件')
    table(['项目','核验后的含义'],[
        ['目标数据','PIDG Rubber Duck基准；本轮没有另行采集物理相机图像。'],
        ['训练观测','120张RGB；120个精确时间、120个相机变换；同时间多相机观测为0。'],
        ['普通RGB训练','原生1600×900；0.8 L1 + 0.2 DSSIM；原RGBA仅用于黑背景合成，不将alpha作新标签。'],
        ['验证','30张val全量原生1600评估；与辅助损失中的held集合不是一回事。'],
        ['固定视角','相机ID86/188；时间编号2/100/200，对应归一时间t=(编号-1)/199。ID不是训练列表索引。'],
        ['统一动态池','15507个动态高斯行，静态库4126点。它们是模型容量，不是鸭子数量。'],
        ['已知与未知','可见性、绑定或边界缺证据即未知，不直接叫背景/动态，不根据暗色或残差删点。']], [138,CW-138])
    h('为什么所有表面在全时段看过，仍不等于完整三维已确定')
    add('不同时间的相机变化与物体运动耦合。需要知道“这次看到的表面”与“上次看到的表面”对应哪个持久节点，并在正确运动补偿后融合。仅有若干二维相似外观，不能自动给出同时间多视角的深度交会或可靠前后表面配对。')
    add('J6提供的位移和D/alpha可以作为假设，却不能再被当作独立真值来证明该假设正确。只有像素误差下降，仍可能留下扁片、拉丝、暗但不透明的支撑。')
    add('训练元数据与val列表已在本报告生成时只读核对；未读取测试集来选训练点。 [S1、S8、S15]','small')

    page(4,'模型来源与公平比较范围','实验谱系')
    note('J6是什么','组合I10动态状态与用户接受的J4静态库，随后比较“静态冻结”与“开放静态opacity/SH”的两臂各2000步。联合J6的val PSNR为31.994673 dB，较匹配控制+0.593549 dB；RGB收益保留，但当时完整性/清理门槛仍未通过。')
    table(['实验','每臂更新','候选相对本轮控制增加了什么','控制条件'],[
        ['L4 时序记忆','2000','跨窗口实测UV记忆及两帧运动位移投影约束','普通RGB'],
        ['L5 表面支撑','2000','连接、局部支撑厚度/面积、观测足迹贡献','RGB + 同一L4时序机制'],
        ['L6 可撤销弹簧','2000','共享固定k=1的局部长度应变','RGB + 同一L4时序机制'],
        ['L7-P / L7-J','0','伪深度准入与跨帧尺度审计','只读数据推断；不是训练臂'],
        ['L8 区间排序','0','绝对标定区间分离准入','没有fit对，未训练'],
        ['L8-R 无尺度排序','600','原始相对逆深度的跨帧稳定顺序','普通RGB']], [88,57,220,CW-365])
    add('L4/L5/L6/L8-R均从同一完整J6开始，保留双Adam、RNG和剩余相机栈。但训练步数及控制损失不同；L5与L6的控制也有不同CUDA数值轨迹。可比较各轮匹配增量，不能把所有候选当同预算、同随机种子重复实验排序。')
    add('上述分支是从J6分别起跑，不是L4训练完再串行接L5、L6、L8-R。原40步CUDA轨迹不确定性失败保留，单次匹配结果没有重复种子置信区间。 [S1-S8]','small')

    page(5,'L4：时序记忆确实约束了什么','跨帧表面记忆')
    h('方法：实际观测记忆，而非生成未见视角')
    add('730条双向RGB片段拼接为641个记忆单元；89条fit重叠连接形成75条长记忆。互为最近、重叠RMS≤0.75px、最大≤1.5px、歧义差≥0.25px；奇数held只检查已选连接。最终22条通过canonical绑定及原生自身可见性，跨度16-20帧。')
    add('候选最多检索4个实际fit关键帧，对共享canonical节点的当前投影、关键帧投影、两帧投影位移联合反传，权重0.01。使用每帧已知相机，因此不是直接把图像位移当世界运动；未观测帧和遮挡间隙不补伪目标。')
    table(['证据','实测','解释'],[
        ['实际覆盖','22行、372观测；时间段356/16/0','偏早期局部记忆；不是全鸭表面积'],
        ['额外时刻','18条记忆共增加根窗口外76个实测观测','增加了信息，但未证明新增完整表面'],
        ['held重投影','J6 0.860543；控制0.895909；候选0.561905 px','已绑定局部对应改善；held仍受普通RGB训练'],
        ['训练实际使用','216/2000步；4734次关键帧对应用','辅助损失不是每步、每表面都有'],
        ['全30val PSNR','控制31.636682；候选31.319321 dB','较控制-0.317361，较J6-0.675352；两门槛失败']], [111,207,CW-318])
    note('能确认与不能确认','能确认局部轨迹约束有效反传、绑定重投影改善。不能确认整物体连通、厚度或离轨迹形状正确；固定86/188仍有碎裂和高alpha拖尾。')
    add('资格每250步刷新，未知暂停、两次矛盾撤销、两次通过恢复。无新Transformer/RNN、无增点或未见表面三角化。 [S2]','small')

    page(6,'L5：连接、厚度和完整性支撑','局部结构约束')
    add('从同J6起点，两臂共用L4时序机制。候选增加四项平均结构损失，总权重0.01，每步最多3组合格表面支撑；不是物体分割或真实体厚监督。')
    table(['分量','实际实现','局限'],[
        ['连接与防塌缩','沿中心连线的2σ椭球间隙；半父长度下限','父参考可能保留旧错误支撑'],
        ['局部支撑厚度','协方差与中心分布最小主轴宽度保持父参照0.8-1.25倍','不是前后表面物理厚度/壁厚'],
        ['切向面积与尺度','两大主轴乘积同区间；单点最大尺度≤1.25倍','不保证封闭或生成缺失面'],
        ['观测足迹贡献','原生排序后的组贡献至少父参照0.8倍','自身可见支撑代理，不是独立GT']], [96,220,CW-316])
    table(['固定360片帧指标','匹配时序控制','L5候选'],[
        ['连接惩罚均值','0.044638','0.018335（约下降58.92%）'],
        ['厚度越界片帧','2','3'],['面积越界片帧','19','13'],['贡献低于0.8倍片帧','96','67'],
        ['全30val PSNR','31.223379','31.859706（+0.636327 dB）']], [214,139,CW-353])
    add('初始60片、275不同动态行、337条片内边应用；起始时段59/1/0，行覆盖约1.77%，不是表面积。候选结构实际使用166步、1174片应用、5527边应用；仍缺少中晚期和未覆盖表面。')
    add('两项RGB不退化门槛通过，但固定86/188整体碎裂和暗高alpha拖尾未一致消除。局部间隙改善不等于完整几何改善；359/360片帧有父贡献，另1个未知不计零误差。 [S3]','small')

    page(7,'L6：弹簧约束带来的收益与代价','共享固定刚度几何正则')
    add('仍从J6起点，与同轮时序控制匹配；不叠加L5厚度/面积/足迹项。每条可靠边的父6个fit帧中位长度作固定参考，统一k=1、总权重0.01。没有质量、接触、外力或材料识别。')
    add('strain = (l(t) - l_ref) / l_ref<br/>E = sqrt(1 + strain²) - 1','formula')
    table(['全初始固定分母的代理','控制','弹簧候选'],[
        ['平均绝对应变','0.108432','0.092034（下降15.12%）'],
        ['平均弹簧能量','0.011789','0.008677（下降26.40%）'],
        ['最负单边应变','-0.411848','-0.695519'],
        ['面积越界片帧','9','16'],['贡献低于0.8倍片帧','67','76'],
        ['val PSNR','31.938599','31.589028（-0.349571 dB）']], [214,139,CW-353])
    h('为什么局部应变变好，整体却没变完整')
    add('平均长度正则主要压低局部伸长，但没有表面生长机制，也不确保面积或每条压缩尾部都改善。连接仍只覆盖60片/275行，起始59/1/0；父中位长度既不是真实静止长度，也可能抑制真实柔性形变。')
    add('178/2000步使用弹簧，1439片、6608边应用。相对J6 PSNR下降0.405645 dB；对照/父模型两项0.3 dB门槛均失败。图像仍有拉丝、碎片和暗支撑，因此没有晋级。')
    note('对“spring-mass会不会有效”的回答','本次已验证的是局部弹簧几何约束，不是完整质量-弹簧动力学。结果不支持直接增加刚度或加入质量/未知外力来弥补对应与覆盖缺陷。 [S4]')

    page(8,'L7-P：伪深度尺度、运动补偿与边界','数据推断审计；0次Gaussian更新')
    add('固定官方Depth Anything V2 Small，对全部120张训练RGB推断；观察400×225，518预处理、FP32 eval。模型不替换、不微调；黑背景也可能有有限预测深度，因此不把其直接解释为表面。')
    h('尺度与运动检验逻辑')
    add('用RGB静止相容轨迹的三角化锚点，拟合每帧正斜率仿射逆深度；空间fit/check隔离和双子集稳定性检查通过后，只允许fit凸包内插值。父J6渲染深度不作尺度拟合GT。')
    add('z(u) = 1 / (a r(u) + b), a &gt; 0<br/>X̂(t) = T_camera(t)<super>-1</super> [z K<super>-1</super> p] - Δx_J6(t)','formula')
    add('相机反投影先消除相机坐标变化，再减去J6时变位移检查同一轨迹的世界一致性。后者仍是父模型条件补偿，不能证明绑定或真实运动正确。偶数拟合、奇数检查，各至少4有效观测。')
    table(['准入指标','实测','原门槛'],[
        ['标定通过','24/120','至少30'],['早/中/晚标定','7 / 7 / 10','每段至少8'],
        ['通过连续运动检查','0条','至少30条，每段至少5'],['不同动态行','0','至少25'],
        ['原生合格轨迹观测','1464中45局部可用','每条fit/check各至少4']], [190,167,CW-357])
    add('单轨最多7/12个有效深度，故0条真正到达运动补偿数值检验。这不能说明“追踪已失效”或“补偿被证明无效”。53196边界观测中42276未知；有限已知子集未显示边界失败更高，不能归因唯一遮挡主因。 [S5]','small')

    page(9,'L7-J：跨帧联合尺度与锚点扩展','数据推断审计；0次Gaussian更新')
    add('新增1981个全图纹理种子，963条RGB合格轨迹、265条静止相容；与原362条合计627条。跨窗重复合并507组，整组隔离fit/check及两份fit；每帧4px去重后5351个锚点观测。不是按蓝色挑静态。')
    add('联合120组归一化逆深度仿射参数，固定Huber误差、0.05二阶时间平滑、正斜率界与极小ridge。平滑是先验，不是尺度正确性证明；held只能检查，不能修系数。')
    img(calibration,'图1：每段40帧。联合与同扩展锚点逐帧方案可比较；原L7-P分割不同，不是严格单因素控制。 [S5-S6]',maxh=205)
    table(['指标','同锚点逐帧','跨帧联合'],[
        ['标定通过/120','26','39'],['原生动态局部观测/1464','16','75'],
        ['到达运动数值检验/122','0','2'],['通过运动检查','0','1（5个fit深度提议）']], [236,121,CW-357])
    add('唯一通过轨迹为052_020、动态行6886，11个局部观测；补偿后held中位残差1.842%、P90 3.623%，仍非独立GT。晚期仅3/40，动态轨迹/行数均未过门槛，5个深度提议没有进入训练。潜在fit凸包平均图像面积6.846%→12.884%也不是准确表面覆盖。 [S6]','small')

    page(10,'L8与L8-R：两个不同的假设','区间失败保留；无尺度候选另行声明')
    table(['设计','L8 区间排序','L8-R 跨帧原始顺序'],[
        ['排序依据','全/分半绝对标定深度区间分离','Small原始相对逆深度patch区间'],
        ['是否依赖公制尺度','依赖稳定标定区间','不拟合绝对尺度'],
        ['候选动态轨迹','L7-J通过metric检验者','原122条T4原生合格轨迹'],
        ['跨帧要求','区间可靠性与空间范围','至少4偶数同号，4奇数验证'],
        ['训练/诊断配对','fit 0/94；held 1/43','fit30、held32；保留全部假设'],
        ['实际训练','0步','两臂各600步']], [104,203,CW-307])
    h('L8为什么没有训练')
    add('94个fit提议中77个绝对区间重叠、17个原始差弱，最终0训练对。唯一held对保留检查用途，未搬入训练。该结果按原门槛保留，不能凭用户允许试训就制造目标。')
    h('L8-R如何独立检验“只要前后关系也行”')
    add('另行预声明无尺度单候选，跳过公制区间但要求原始排序在观察窗口内持续、局部平滑、跨fit/check时间一致；没有回改L8失败结果，也没有放宽旧全局几何准入。用户授权的是局部探索性试验，不是自动推广到整鸭关系。')
    note('身份与顺序仍是两件事','关系端点是“已绑定动态行的实测UV”与“静止相容RGB锚点UV”。两端并不带鸭子编号或环境语义；最多4对/图不是物体数量。 [S7-S8]')

    page(11,'L8-R：目标构造与损失公式','当前已完成训练的核心机制')
    add('令S为该图原始相对逆深度q90-q10。两端5×5 patch自身范围≤0.1S、像素相距≥8，且patch区间间隔≥0.1S才得到强符号。较大逆深度表示较近；缺失/边界/矛盾保留未知。')
    add('同一动态轨迹/锚点组的12帧中，至少4个偶数fit支持同一符号s；至少4个奇数check支持该符号。check只能拒绝，不能改选符号。整锚点组隔离，稳定track/group排序后每图最多4对，置信度固定0.5。')
    table(['完整构造分母','fit库','诊断held库'],[
        ['组级假设','4540','2953'],['跨时间支持通过','41','27'],['逐帧提议（上限前）','193','129'],
        ['最终固定关系对','30','32'],['动态行','6498、15419','6498、15419'],['帧索引','100-114中的8个偶数','101-115中的8个奇数']], [205,153,CW-358])
    add('在原生400联合渲染中，双线性采样D和alpha后令z=D/alpha；先检查正/有限深度、alpha&gt;0.5及动态行当前重投影≤4px。两端深度的有向log差为：')
    add('g = s [log(z_anchor) - log(z_dynamic)]<br/>e = max(0, log(1.02) - g)<br/>L_order = (0.001 / N) Σ [0.5 · valid · (sqrt(1 + (e/0.05)²) - 1)]','formula')
    add('N是该帧原提议固定分母，不随当前可用子集缩小。监督符号/置信度/资格停止梯度；梯度只经过当前原生D/alpha回到动态canonical/grid/deform。s=+1表示动态端应更近；没有指定绝对深度或大的深度间隔。 [S8-S9]','small')

    page(12,'资格可撤销，但有效监督很稀疏','实际训练过程与使用量')
    img(controller,'图2：每100步复核固定30对关系。初始19活跃/11暂停；候选在500步刷新后7活跃、2暂停、21撤销。末态沿用该次资格，非600步额外复核。 [S10]',maxh=224)
    table(['检查/训练项目','固定规则或实测'],[
        ['自身贡献与RGB','原生单次全场深度排序，自身行patch贡献份额≥0.01；动态端UV的观测/渲染patch NCC≥0.6，二者标准差≥2（255标度）'],
        ['撤销/恢复','未知立即暂停；2次矛盾撤销；2次通过恢复；每步另做当前投影及alpha/正深度防护'],
        ['训练预算','两臂各600个1600 RGB更新；仅候选额外400顺序渲染；固定epsilon 1e-15'],
        ['实际顺序使用','20/600步产生非零辅助损失；74次有效pair观测；分布在102/104/106/112索引'],
        ['没有变化的状态','静态模型及Adam、physics/Adam/图缓存、拓扑均冻结；没有叠加L4/L5/L6损失']], [130,CW-130])
    add('规则中的1%自身贡献仅是可见性准入，不代表该行主导联合深度。光线上的其他高斯仍能显著影响D/alpha，这是“图像顺序”到“具体表面节点关系”的关键局限。')
    add('20/600约3.33%；这是有效应用量，不可将600个RGB更新误写成600次深度约束更新。损失和门槛固定，未在结果后自动提高权重或加训。 [S8-S10]','small')

    page(13,'数值比较：关系一致数没有增加','匹配结果与跨轮背景')
    table(['模型','PSNR dB','SSIM','L1'],[[name,f"{means[key]['PSNR']:.6f}",f"{means[key]['SSIM']:.6f}",f"{means[key]['L1']:.8f}"]
        for key,name in [('J6_parent','J6父'),('rgb_control','本轮RGB控制'),('ordinal_depth','本轮相对顺序')]], [158,112,112,CW-382])
    table(['顺序诊断（非GT）','J6','RGB控制','候选'],[
        ['fit一致/固定30对','9/30','9/30','9/30'],['fit有效/固定30对','28/30','28/30','28/30'],
        ['held一致/固定32对','19/32','19/32','19/32'],['held有效/固定32对','32/32','32/32','32/32']], [220,97,97,CW-414])
    add('候选较本轮控制-0.100921 dB，较J6-0.164675 dB，两项最大下降0.3 dB门槛通过，但只是“不明显退化”。fit平均有向log间隔-0.026816→-0.024436，held 0.022797→0.024123；没有配对改变先验顺序侧或有效性。')
    img(comparison,'图3：各轮候选与各自匹配控制。L4/L5/L6各2000步，L8-R600步；不可将这些点当重复种子或同一控制的排行。 [S2-S4、S8]',maxh=205)
    add('全部122轨迹/1464观测：候选held重投影0.911032px，对照0.909937px，J6 0.888095px；没有整体绑定改善。held只避开辅助目标，仍接受普通RGB，不是独立几何泛化。 [S8]','small')

    page(14,'固定相机86：早、中、晚比较','原生1600图像的同区域裁切')
    img(fixed[86],'图4：列=J6 / 本轮纯RGB控制 / 相对顺序候选；行=时间编号2 / 100 / 200。各格为原生RGB相同裁切，不重绘、不修图。 [S11]',maxh=465)
    h('观察')
    add('早期表面仍松散、缺口明显；中期可辨轮廓并不代表后侧颗粒串消失。晚期三模型都保留明显拉长颗粒片、破碎区域和暗支撑。候选没有将碎片稳定连成完整物体。')
    add('静态层完全冻结，蓝色支撑结构相同。动态变化造成的联合遮挡不同，不应称为本轮静态重建提升。固定相机与跨时间组合无独立RGB/几何GT，图像用于缺陷比较而非精确误差标定。','small')

    page(15,'固定相机188：仍有拉丝与缺口','原生1600图像的同区域裁切')
    img(fixed[188],'图5：列=J6 / 本轮纯RGB控制 / 相对顺序候选；行=时间编号2 / 100 / 200。与图4相同取样；各相机内部裁切固定。 [S11]',maxh=465)
    h('观察')
    add('早期鸭体仍有碎裂，中期出现左右扩展的粒子带与拉丝；晚期RGB较接近可辨鸭形，但不能说明暗处的高alpha结构已经消失。三模型主要差异是局部细节，并无一致补全。')
    add('“局部关系没有改变”与“画面整体相似”是本轮共同证据；它们不证明所有潜在相对先验无效，也不证明RGB数据不可解。当前监督只触及晚期2个动态高斯行。','small')

    page(16,'RGB变暗不等于拖影已消除','动态RGB、alpha与D/alpha联合复核')
    img(decomposition,'图6：相对顺序候选。上行c188/t100，下行c86/t200；列=动态RGB / 动态alpha / 动态D/alpha。深度色标复用各相机同一父静态范围，黑色低alpha为缺失，不是零误差。 [S11]',maxh=332)
    add('RGB中偏暗的侧向条带，在alpha和D/alpha仍有明显支撑。观测到“颜色更黑”不能据此宣布删除了错误几何；反之，高alpha也不能单凭颜色被判为背景或直接删点。')
    table(['六固定图暗RGB且alpha&gt;0.5像素总计','像素数','相对关系'],[
        ['J6','107498','父模型参照'],['本轮RGB控制','121627','比J6多'],['相对顺序候选','113170','较控制少6.95%，仍较J6多5.28%']], [CW-223,99,124])
    add('定义为动态渲染各RGB通道&lt;0.05且alpha&gt;0.5，仅在评估中统计；分母每图1440000像素，共8640000像素应用。不是独立GT面积、不是按颜色给训练分类。视觉清理与完整性门槛均失败。 [S8、S11]','small')

    page(17,'相关训练分支的同视角横向查看','机制不同，缺陷却未稳定消失')
    img(historical,'图7：相机188/t100相同世界区域。列=J6、L4、L5、L6、L8-R；上行RGB，下行动态alpha。各候选预算及损失不同，不作严格同预算优劣排名。 [S2-S4、S8、S11]',maxh=255)
    h('跨轮共同现象')
    add('时序、局部结构、弹簧和相对深度各自改变局部细节，但表面碎裂、侧向高alpha支撑仍在。它们优化的代理不同：像素重投影、父几何支撑、父中位边长、单目先验顺序；任何一项自洽都不足以验证整鸭表面。')
    img(observed,'图8：训练列表index119的原始观测及三模型原轨迹渲染，从已有原图重排。此处外观更整洁；这是训练相机/时间观测，不是固定86/188跨时间的独立GT。 [S11]',maxh=140)
    add('图7展示的是候选，不含各轮匹配控制；数值因果比较应回到第13页各轮控制增量。图8与图7时间/相机条件不同，用于说明评价范围差异，不能相减求几何误差。','small')

    page(18,'目前缺的是什么，下一步如何检验','技术解释与可执行方向')
    table(['证据支持的限制','不能过度推断的结论','优先检验方向'],[
        ['可靠支撑时间覆盖不足：L4/L5/L6偏早，L8-R仅晚期2行','不能据此断言整段RGB没有其他可用表面','分时间、视角、可见表面列固定分母，扩大可靠绑定再训练'],
        ['二维轨迹到canonical/局部片的对应有不确定性','不能把行数、图连接数当物体身份','保持可撤销连接；独立检验遮挡前后与表面归属'],
        ['联合D/alpha混合多高斯贡献，自身份额下限仅1%','排序梯度不保证修改目标表面节点','显式报告贡献主导性及表面关联，再设计关系损失'],
        ['父参考长度/厚度/深度可能保存错误支撑','不能由自身代理变好证明真实几何','增加互斥观测验证；只保护有证据的表面关系'],
        ['当前无新增表面生长或可靠前后表面配对','更多正则不能自动创造缺失表面','若要生长/厚度约束，先提出可验证的新表面与厚度证据']], [174,160,CW-334])
    h('建议的下一试验次序，不是已执行结果')
    add('先只读核验更广时间/视角的表面节点和可见贡献；保持未知及原始候选。证据足够后，固定一个节点级相对关系或局部表面融合机制，声明匹配预算/阈值/原生梯度与恢复验收，再进行一轮训练。不要同时叠加多个损失，避免无法归因。')
    note('当前决策','保留J6，L8-R完整权重保存但不晋级；无自动加训、提高损失权重、扫参或改门槛。此次负结果不足以否定时序信息、所有伪深度或RGB可解性。真实质量/本构/外力不是本轮新增内容。')
    add('限制判断来自固定分母、原生图像与代码机制，未使用val/test选择新训练点。 [S1-S11]','small')

    page(19,'工程核验、失败留档与可复现性','正确运行与几何成功是两条验收线')
    table(['核验项目','本轮实际完成'],[
        ['测试与默认关闭','20项相关测试；原生RGB/alpha/D精确同值，原梯度容差通过'],
        ['恢复方式','用户已接受：完整模型、双Adam及绑定、RNG、相机栈、physics图精确恢复；同一份原生梯度的Adam更新精确'],
        ['梯度','实际顺序损失可计算；资格专用强制违例验证canonical/grid/deform有限非零梯度，不进入训练目标'],
        ['全程重放','1200相机采样，10个100-500步已存刷新模型原生逐值重放，独立资格状态转移核验'],
        ['产物核验','180图SHA/可读性；6完整Gaussian PLY全部字段、9世界XYZ诊断；两H264各200帧完整解码'],
        ['冻结与收尾','608个冻结文件SHA、17个任务冻结Git原始字节；状态pending:false，任务锁已释放']], [127,CW-127])
    h('第一次100步失败及独立修复')
    add('首个正式RGB对照完成100步后，刷新状态中的旧measurement元数据与新字段重复合并导致退出。当时没有完整100步checkpoint，故不从部分状态续接；失败stdout、初始状态、进度保留。独立repair_run仅剥离旧测量元数据，再调用冻结原转移逻辑。')
    add('补测超过600次状态调用与撤销/恢复，重新通过两臂短训和独立恢复后，从原完整J6重跑各600步。保留比较共1200更新，另有失败控制100步及两轮资格短训；不混入最终模型步数。原epsilon1e-15、损失、门槛及历史40步CUDA失败均保留。')
    add('关键提交：5e1a0800 预声明；aa62478d 首次资格；6b2d6f3c 修复留档；4a8173d8 修复后资格；eb70f4e4 训练完成；53b7fbfc 结果；9874a28d 最终验证。浏览器播放未实测；人工看的是原图与解码接触图，并非人工逐帧播放。 [S8-S14]','small')

    page(20,'来源、权重与复核入口','仓库内证据索引')
    add('以下路径均相对于 F:/A_Edinburgh/PhysGauss。主报告来源与所有引用图片的SHA记录在同目录manifest.json；本PDF不改变历史实验文件。','small')
    refs=[
        ('S1','J6方法、来源与结果','RubberDuck_I10_joint/v1/assessment.md'),
        ('S2','L4时序方法与结果','RubberDuck_temporal_memory/v1/assessment.md'),
        ('S3','L5结构方法与结果','RubberDuck_surface_support/v1/assessment.md'),
        ('S4','L6弹簧方法与结果','RubberDuck_revocable_springs/v1/assessment.md'),
        ('S5','L7-P伪深度准入','RubberDuck_pseudo_depth/v1/assessment.md'),
        ('S6','L7-J联合尺度','RubberDuck_joint_scale/v1/assessment.md'),
        ('S7','L8区间0fit失败','RubberDuck_ordinal_depth/v1/assessment.md'),
        ('S8','L8-R报告/协议/评估','RubberDuck_relative_order/{protocol.json,v1/assessment.md,v1/evaluation.json}'),
        ('S9','排序实现','variants/staged_reconstruction/{i10_relative_order,i10_ordinal_depth}/core.py'),
        ('S10','使用量与刷新审计','RubberDuck_relative_order/v1/{controller_audit.json,pairs_audit.json}'),
        ('S11','原生图片、PLY和视频','output/staged_reconstruction/RubberDuck_{relative_order,temporal_memory,surface_support,revocable_springs}/v1/'),
        ('S12','独立产物审计','RubberDuck_relative_order/v1/independent_audit.json'),
        ('S13','恢复与原生梯度','RubberDuck_relative_order/v1/{recovery.json,native_check.json}'),
        ('S14','最终状态/验证/失败说明','RubberDuck_relative_order/{refresh_repair.md,v1/final_review.json,v1/final_validation.json}'),
        ('S15','训练/验证相机元数据','data/derived/kinematic_resplit/{training/transforms_train.json,transforms_val.json}')]
    for sid,label,path in refs:
        prefix='' if path.startswith(('variants/','output/','data/')) else 'experiments/staged_reconstruction/'
        add(f'<b>[{sid}] {label}</b>　'+escape(prefix+path),'cap')
    h('当前权重 SHA-256')
    for label,value in [('J6',protocol['parent_sha256']),('本轮RGB控制',ev['checkpoint_audits']['rgb_control']['checkpoint_sha256']),
                        ('本轮顺序候选',ev['checkpoint_audits']['ordinal_depth']['checkpoint_sha256'])]:
        add(f'{label}: {value[:32]}<br/>{value[32:]}','small')
    add('两个候选checkpoint位于 runs/pidg_staged_reconstruction/RubberDuck_relative_order/v1/formal/{rgb_control,ordinal_depth}/checkpoint.pt。保留完整状态，不自动恢复或重跑。','cap')

    doc=ReportDoc(str(PDF),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=44,
        title='Rubber Duck重建实验比较技术报告',author='PhysGauss实验记录',subject='时序、局部结构、弹簧、伪深度与相对顺序的证据比较',pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(str(PDF));texts=[q.extract_text() for q in reader.pages]
    assert len(reader.pages)==20,(len(reader.pages),page_specs)
    assert len(reader.outline)==20
    for i,(_,title) in enumerate(page_specs):assert title.replace('<br/>','')[:6] in texts[i].replace('\n',''),(i,title,texts[i][:150])
    assert all(len(t)>80 for t in texts)
    for token in ['31.829998','31.930919','31.994673','9/30','19/32','608','9874a28d']:assert token in '\n'.join(texts),token
    manifest=dict(pdf=str(PDF.relative_to(ROOT)),sha256=digest(PDF),pages=len(reader.pages),
        authoring_source=Path(__file__).relative_to(ROOT).as_posix(),authoring_source_sha256=digest(__file__),
        experiment_files_and_images=sources,
        figures={name:dict(embedded_in_pdf=True,sha256=digest(path)) for name,path in figures.items()},expected_titles=page_specs,
        no_training_or_frozen_result_mutation=True,language='zh-CN',font_embedded='Microsoft YaHei TrueType regular/bold',
        evidence_scope='J6, L4, L5, L6, L7-P, L7-J, L8 and L8-R; no new experiments')
    (DEST/'RubberDuck_实验比较技术报告_20261106.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PDF_CREATED',PDF,'PAGES',len(reader.pages),'SOURCE_FILES',len(sources),flush=True)


if __name__=='__main__':build()
