"""Chinese technical analysis of completed C1 integrity and ObjectID ablations."""
from datetime import datetime,timezone
from html import unescape
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
import pidg_ablation_report_layout as L

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf';A=OUT/'integrity_analysis_assets'
PDF=OUT/'PIDG_Completeness_Technical_Analysis_CN.pdf'
RECORD=[];PAGES=15


def plain(text):
    text=re.sub(r'<super>(.*?)</super>',r'^(\1)',text)
    text=re.sub(r'<sub>(.*?)</sub>',r'_(\1)',text)
    return unescape(re.sub('<[^>]+>','',text.replace('<br/>','\n').replace('<b>','**').replace('</b>','**')))


class Page(L.Page):
    def __init__(self,n,label,title,subtitle=''):
        RECORD.extend(['','## '+str(n)+'. '+title]);super().__init__(n,label,title,subtitle)
    def text(self,text,style='body',gap=10):
        RECORD.append(plain(text));super().text(text,style,gap)
    def heading(self,text):RECORD.extend(['','### '+text]);super().heading(text)
    def callout(self,text):RECORD.append('> '+plain(text));super().callout(text)
    def table(self,data,widths,highlight=None,size='table'):
        cell=lambda v:str(v).replace('|','\\|').replace('\n','<br>')
        RECORD.extend(['','| '+' | '.join(map(cell,data[0]))+' |','| '+' | '.join(['---']*len(data[0]))+' |'])
        RECORD.extend('| '+' | '.join(map(cell,row))+' |' for row in data[1:]);super().table(data,widths,highlight,size)
    def image(self,path,height,caption=None):
        RECORD.append(f'![{caption or Path(path).stem}](integrity_analysis_assets/{Path(path).name})')
        super().image(path,height,caption)


def build():
    data=json.loads((A/'data.json').read_text(encoding='utf-8'));ms=data['models'];m={x['id']:x for x in ms}
    b,h=m['C1'],m['H3'];delta=data['pairwise_vs_C1']['H3']
    for name,file in [('CN','msyh.ttc'),('CN-Bold','msyhbd.ttc'),('Arial','arial.ttf'),('PathFont','consola.ttf')]:
        pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file,subfontIndex=0))
    pdfmetrics.registerFontFamily('CN',normal='CN',bold='CN-Bold',italic='CN',boldItalic='CN-Bold')
    L.TOTAL_PAGES=PAGES;L.REPORT_TIME=datetime.now(timezone.utc).astimezone(ZoneInfo('Europe/London'));L.page_records.clear()
    L.c=canvas.Canvas(str(PDF),pagesize=(L.W,L.H),pageCompression=1)
    L.c.setTitle('PIDG 完整度与 ObjectID 实验技术分析');L.c.setAuthor('PIDG Extension Project')
    L.c.setSubject('Rubber Duck; final50k; C1/H1-H4/F1-F2/O1-O2; fixed50 test')
    pct=lambda obj,k:100*obj['integrity']['mean'][k]
    score=lambda obj,k='integrity_score':obj['integrity']['mean'][k]
    path=lambda value:L.path_markup(str(value).replace('/','\\'))
    p=Page(1,'结论 / EXECUTIVE FINDING','PIDG 已训练模型技术分析','Rubber Duck | 完整度、外观与运动约束 | 数据核验于 '+L.REPORT_TIME.strftime('%Y-%m-%d %H:%M %Z'))
    p.callout('当前综合最优为 <b>H3 = C1 + 平衡覆盖损失 + 弱邻域时间应变约束</b>。在相同 50 张测试图、final 50k 快照下，H3 的 PSNR、SSIM、LPIPS、MS-SSIM 均优于 C1，平均完整度评分也更低。')
    p.table([['指标','C1 对照','H3','变化'],['PSNR ↑',f"{b['test']['PSNR']:.4f}",f"{h['test']['PSNR']:.4f}",'+0.5061 dB'],
        ['LPIPS ↓',f"{b['test']['LPIPS']:.6f}",f"{h['test']['LPIPS']:.6f}",'-9.55%'],
        ['内部低覆盖 α<0.9 ↓',f"{pct(b,'core_low_90'):.5f}%",f"{pct(h,'core_low_90'):.5f}%",'-33.24%'],
        ['背景误覆盖 ↓',f"{pct(b,'background_false_coverage'):.4f}%",f"{pct(h,'background_false_coverage'):.4f}%",'-19.97%'],
        ['平均完整度评分 ↓',f'{score(b):.6f}',f'{score(h):.6f}','-11.30%']], [155,112,112,128],highlight=3)
    p.heading('如何使用这个结论')
    p.text('以 H3 作为下一轮默认模型，保留 C1 作为稳定对照。F1 作为严格低覆盖和碎片损失的专项候选保留；H1 的前景漏覆盖与完整度评分 P95 更好，但外观和背景覆盖退化，不作为整体替代。')
    p.text('H3 并非逐帧最优：PSNR 改善 33/50 帧，下降 17/50 帧。第 85 帧下降 6.24 dB，仍有明显拖影。H3 的测试碎片损失也略高于 C1，验证集的严格低覆盖率未改善。')
    p.text('本报告主表分析 C1、H1-H4、F1-F2 七组完整度实验；另评估 O1/O2 ObjectID，并回顾 C2/C3 颜色修正、材质、中心与 iso 消融。早期训练协议不同的结果不混入同一排名。','small')
    p.finish()

    p=Page(2,'协议 / PROVENANCE','比较是否公平：先固定口径')
    p.table([['项目','本报告采用的口径'],['数据与任务','Rubber Duck；200 个原始时刻；120 train / 30 validation / 50 test。三者帧号集合两两无交集，合并覆盖 1-200。'],
        ['测试时刻','固定官方 50 张：原始帧号 1, 5, 9, ..., 197；按原始帧号排列。'],['时间与图像','t=(frame_id-1)/199，训练/验证/测试及物理时间共享；1600×900，量化 PNG 后评估。'],
        ['训练预算','七组均为总计 50,000 iter，seed=0，width=256、depth=0，无 residual adapter。'],['初始状态','七组共享完整 3k 检查点；含 Gaussian、网络、优化器及随机状态。C1 直接复用，未重复训练。'],
        ['续训的含义','H1/H4/F1/F2 从各自 20k 全状态恢复至总计 50k；不是在一个已训练 50k 模型上再加 50k。'],['模型选择','验证集选择各指标最佳快照。主表统一报告 final50k，不按测试结果混选不同 iter。']], [99,408])
    p.heading('不能外推的部分')
    p.text('时刻集合互不重叠，但它们在同一段时间轴交错分布，因此验证的是同场景时间插值与已给相机条件下的重建；不是连续未来时间段预测，也不是新场景泛化。')
    p.text('测试 RGB/alpha 不进入本轮梯度损失，训练损失只使用训练帧。验证图像用于超参、筛选和快照选择。该官方测试集此前已被查看和比较，历史暴露不会因重新划分而消失；本结果不是全新盲测。')
    p.text('只有一个场景和一个随机种子；CUDA 存在非确定性。共享起点提高可比性，但不能替代多随机种子重复实验或证明统计显著性。','small')
    p.finish()

    p=Page(3,'结构 / ARCHITECTURE','C1 是什么：共同网络与新增项')
    p.text('本报告的 C1 指 <b>C1_repaired</b>，不是未经修改的原始 PIDG。它保留 PIDG 的 canonical Gaussian、时空形变场以及速度/应力和动量残差路径，并继承 V3 的全局时间与图像目标。')
    p.table([['阶段','共同结构'],['编码','XYZ 哈希：16×2=32 维；XYT/YZT/XZT 三个哈希编码：各 32×2，总计 192 维。'],['融合','32→256 Linear→Sigmoid→[-1,1] 空间门控，与 192→256 Linear→ReLU 时间分支逐元素相乘。'],['输出','共享额外隐藏层 depth=0；三个 Linear 头：256→7（warp 四元数+平移）、256→4（旋转增量）、256→3（尺度增量）。'],['参数解释','上述形变 MLP 含偏置共 61,454 参数，不包含哈希表、Gaussian 参数库、速度/应力网络。depth=0 不表示没有 MLP。'],['C1 修正包','SH 辐射使用 Softplus(β=5)，逆变换初始化 DC；L1+DSSIM、LPIPS=0.02；mask L1=0.03、前景 RGB L1=0.1，从 3k 起用 2k 渐增。']], [85,422])
    p.table([['变体','在 C1 上额外增加','最终点数'],['C1','无本轮新增项',str(b['points'])],['H1','平衡前景覆盖 + 背景抑制',str(m['H1']['points'])],['H2','弱邻域时间应变一致性',str(m['H2']['points'])],['H3','H1 + H2',str(h['points'])],['H4','颜色一致的覆盖定向增密',str(m['H4']['points'])],['F1','H3 + 已覆盖前景难像素 RGB 项',str(m['F1']['points'])],['F2','F1 的三项新权重全部减半',str(m['F2']['points'])]], [55,350,102],highlight=4)
    p.text('H/F 七组没有增大 MLP，也没有加入新的 E 材料参数、ObjectID、iso、速度导数一致性或中心约束。训练目标改变可影响 15k 以前的常规增密，所以最终点数不完全一致。','small')
    p.finish()

    p=Page(4,'机制 / COVERAGE + COHESION','H1 / H2：二维覆盖与运动连续性')
    p.heading('H1：对内部缺口和背景溢出分别归一化')
    p.text('令 M 为训练 RGBA 的 GT alpha，a 为渲染 alpha；C 为 7×7 腐蚀后 alpha&gt;0.95 的内部区域，B={M≤0.5}。对前景/背景分别求均值，避免大量背景像素稀释内部监督。')
    p.callout('L<sub>cov</sub> = mean<sub>p∈C</sub> ReLU(0.95-a<sub>p</sub>) + 0.25 mean<sub>p∈B</sub> a<sub>p</sub><br/>权重 0.002；从 3k 开始，4k 渐增至满权重。')
    p.text('它比单一全图 mask L1 更直接处理内部低透明度，并显式抑制背景覆盖。梯度可改变 Gaussian 位置、尺度、不透明度及其形变；并未单独冻结几何，因此必须同时观察颜色与边界。')
    p.heading('H2：相邻训练时刻的邻域距离变化约束')
    p.text('每次采样 96 个锚点，每点 8 个邻居，canonical 半径 0.05。只保留双方 opacity&gt;0.05、DC 颜色距离&lt;0.25，且在首/中/末训练时刻始终相距 10<super>-5</super> 至 0.15 的边。无需真值 object ID，但这些筛选并不等同于正确实例分割。')
    p.callout('d<sub>ij</sub>(t)=||x<sub>i</sub>(t)-x<sub>j</sub>(t)||<br/>e<sub>ij</sub>=ReLU(|log d<sub>ij</sub>(t′)-log d<sub>ij</sub>(t)|-log(1+ε))<br/>L<sub>strain</sub>=mean Huber<sub>β=0.1</sub>(e<sub>ij</sub>)<br/>ε=min(0.2, 0.03×|t′-t|×199)，t′ 为相邻训练时刻。')
    p.text('权重 10<super>-4</super>，8k 开始、4k 渐增；每 20 iter 计算并乘 20 补偿。canonical 点位置和筛边过程 detach，梯度主要作用于形变轨迹。距离式约束允许刚体平移/旋转，并为真实拉伸保留容差；它不是强刚性 ARAP，也不是完整应变本构。','small')
    p.finish()

    p=Page(5,'机制 / DENSITY + FEEDBACK','H4 / F1 / F2：为什么更复杂未必更好')
    p.heading('H4：覆盖驱动、颜色过滤的晚期克隆')
    p.text('只在 C1 上增密，不含 H1/H2。16k-36k 累计训练时刻的内部覆盖缺口：(0.95-a)<super>+</super>×C×[a&lt;0.7]；按实际渲染足迹找到已有贡献点，每 2k 决策一次。候选须在至少 3 个不同训练时刻出现，训练 RGB 标准差每通道≤0.15。')
    p.text('每窗口最多补 256 点，保留父点。新点颜色用缺口处训练 RGB 加权平均经 Softplus 逆变换初始化，高阶 SH 清零；初始 opacity 限制在 0.005-0.05。偏移标准差为 0.25×min(scale,0.02)。')
    p.callout('实际执行 10 个窗口，共新增 <b>363 点</b>；final 84,607 点。完全没有贡献点的空射线不能提供 3D 种子；“补点”不等于找到了正确深度或物体归属。')
    p.heading('F1：H3 上增强已覆盖前景的困难颜色像素')
    p.text('先用 GT 前景与 a≥0.8 的交集筛像素，再取 RGB L1 最大的 20% 求均值，权重 0.01。8k 起用 4k 渐增，每 4 iter 计算并补偿。alpha 筛选 detach，但 RGB 损失仍可反传到几何、尺度与 opacity，不能称为“只改颜色”。')
    p.text('F1 从共同 3k 起点按反馈配置训练，并非加载 H3 的 50k 权重续训。F2 将 coverage、strain、tail 权重分别降至 0.001、5×10<super>-5</super>、0.005；F1/F2 均未启用晚期增密。')
    p.heading('应如何解释失败')
    p.text('增密可能复制已有位置/深度误差并扩大轮廓；困难像素重加权可能改变整体外观与几何之间的平衡；降低权重也会改变优化轨迹。这些是与结果相符的机制假设，当前日志并不能定位唯一因果，更不能证明所有增密或颜色修正都无效。')
    p.finish()

    p=Page(6,'结果 / APPEARANCE','四项图像指标：H3 同时占优','固定 50 张测试图；各图指标先计算，再对 50 张求均值。↑ 越高越好，↓ 越低越好。')
    rows=[['模型','PSNR ↑','Δ dB','SSIM ↑','LPIPS ↓','MS-SSIM ↑']]
    for x in ms:rows.append([x['id'],f"{x['test']['PSNR']:.4f}",f"{x['test']['PSNR']-b['test']['PSNR']:+.4f}",f"{x['test']['SSIM']:.7f}",f"{x['test']['LPIPS']:.7f}",f"{x['test']['MS-SSIM']:.7f}"])
    p.table(rows,[45,78,70,104,104,106],highlight=4)
    p.image(A/'quality.png',186,'图 1：相对 C1 的 PSNR 与 LPIPS 变化。H3 的两项变化方向都更有利。')
    p.text('H3 的 SSIM 比 C1 提升 0.0003886，MS-SSIM 提升 0.0003985。H2 虽然 PSNR 下降约 0.459 dB，但 LPIPS 与 MS-SSIM 略好于 C1，说明不同指标并非完全等价。')
    p.text('H1、H4、F1、F2 没有同时改善四项指标。尤其 H4 的 PSNR 下降约 1.005 dB，不能仅因它执行了增密就判定效果更好。主表没有把其他 iter 的单项最佳指标拼成一个不存在的模型。')
    p.text('本报告不将早期 150 train / test 参与选优、旧时间口径或额外续训预算下的 A/T/W 实验与此表直接排名；它们的差值不能用于归因本轮模块效果。','small')
    p.finish()

    p=Page(7,'指标 / DEFINITIONS','“完整度”测量的到底是什么')
    p.text('以下指标均基于 <b>2D 渲染 alpha 与 GT alpha</b>；不是 3D 内部体积、封闭性、碰撞或实例分割的真值。M、a∈[0,1]；F={M&gt;0.5}，B={M≤0.5}，P={a&gt;0.5}。C=minimum_filter<sub>7×7</sub>(M)&gt;0.95，图外填 0，相当于向内排除约 3 像素边界。')
    p.table([['指标','单帧定义（每帧求比例，再平均 50 帧）'],['内部低覆盖 r50 / r90','|C ∩ {a<0.5 或 a<0.9}| / |C|。越低，GT 内部达到对应 alpha 的像素越多。不是删除某模块后剩余区域。'],['前景漏覆盖 m50','|F ∩ {a≤0.5}| / |F|。包含轮廓附近；与腐蚀后的内部低覆盖不同。'],['背景误覆盖 b','|B ∩ P| / |B|。分母是 GT 背景，不是预测前景面积。'],['Mask IoU','|F ∩ P| / |F ∪ P|；越高越好。'],['不透明外观错误 o','|F ∩ {a≥0.9} ∩ {mean_RGB |I-G|>0.15}| / |F|。用来发现“alpha 填满但颜色不对”。'],['碎片损失 f','对每个面积≥32 像素的 GT 前景连通块，求 1-最大有效预测连通块面积/GT 块面积；再按块、按帧平均。']], [110,397])
    p.text('碎片连通使用 4 邻接；预测连通块在 GT 块内求取，并去除面积&lt;max(8, floor(0.001×GT 块面积)) 的小块。因此 f 同时受漏覆盖与碎裂影响，不等于真实物体实例碎片数；接触物体可能在 GT 上属于同一块。','small')
    p.callout('完整度评分 S = 0.45r90 + 0.25m50 + 0.20f + 0.10o + 0.10b<br/>所有项用 0-1 比例；该加权和为项目自定义诊断，非标准 benchmark 分数，权重和为 1.10，并非概率。越低越好。')
    p.text('表中带 % 的列为比例×100；百分点差与相对百分比下降必须区分。低覆盖接近零只表示超过 alpha 阈值，不能保证颜色、深度、法线或运动都正确。','small')
    p.finish()

    p=Page(8,'结果 / COMPLETENESS','完整度收益与单项取舍')
    rows=[['模型','内部 r50 % ↓','内部 r90 % ↓','背景 b % ↓','IoU ↑','评分 S ↓']]
    for x in ms:rows.append([x['id'],f"{pct(x,'core_low_50'):.5f}",f"{pct(x,'core_low_90'):.5f}",f"{pct(x,'background_false_coverage'):.4f}",f"{score(x,'mask_iou'):.5f}",f"{score(x):.6f}"])
    p.table(rows,[45,98,98,96,82,88],highlight=4)
    p.table([['专项诊断 ↓','C1','H3','F1','当前最优'],['碎片损失 f',f"{score(b,'fragment_loss'):.6f}",f"{score(h,'fragment_loss'):.6f}",f"{score(m['F1'],'fragment_loss'):.6f}",'F1'],
        ['最差帧评分',f"{score(b,'worst_score'):.6f}",f"{score(h,'worst_score'):.6f}",f"{score(m['F1'],'worst_score'):.6f}",'F1'],
        ['评分 P95',f"{score(b,'p95_score'):.6f}",f"{score(h,'p95_score'):.6f}",f"{score(m['F1'],'p95_score'):.6f}",'H1: 0.007353'],
        ['前景漏覆盖 %',f"{pct(b,'foreground_missing_50'):.5f}",f"{pct(h,'foreground_missing_50'):.5f}",f"{pct(m['F1'],'foreground_missing_50'):.5f}",'H1: 0.22717%']], [115,89,89,89,125])
    p.text('H3 的 r50 相对 C1 降低 66.06%，r90 降低 33.24%，背景误覆盖降低 19.97%；IoU 从 0.91610 升至 0.93148。其平均评分、前景 RGB L1 和不透明外观错误也为七组最佳。')
    p.text('但 H3 的碎片损失 0.006637 略高于 C1 的 0.006564；平均 extra_fragments 为 0.010，对照为 0.005。F1 的严格 r50 最低，但与 H3 仅差约 0.000108 个百分点，背景误覆盖却从 H3 的 0.8036% 升至 1.2820%，PSNR 下降约 1.097 dB。')
    p.callout('综合最优选 H3；若唯一目标是最差帧完整度评分或碎片损失，可保留 F1。不能因为一个阈值指标更接近 0，就忽略颜色、边界和其余时刻。')
    p.finish()

    p=Page(9,'结果 / TIME DISTRIBUTION','按原始帧号分析，而非按 PSNR 排序')
    p.image(A/'time.png',272,'图 2：上图为逐帧 PSNR，下图为 H3-C1 差值；横轴是整个序列的原始帧号。')
    p.table([['测试帧段','改善帧 / 总帧','平均 ΔPSNR'],*[[f"{x['frame_range'][0]}-{x['frame_range'][1]}",f"{x['improved']} / {x['count']}",f"{x['mean_delta']:+.4f} dB"] for x in delta['thirds']]], [180,160,167])
    p.text(f"H3 的平均差值 {delta['mean']:+.4f} dB，中位数 {delta['median']:+.4f} dB，33/50 帧改善。因此提升并非只由一个极端正例产生，但末段收益较小，仍存在局部严重失败。")
    p.text('最大增益：帧 73，C1 34.6570 → H3 39.1782 dB；最大退化：帧 85，C1 36.8655 → H3 30.6273 dB。这两个时刻均为测试图，下面只用于事后诊断，不据此调参后继续把它们当作盲测。')
    p.text('完整逐帧 CSV 保留七组 PSNR 及 H3-C1 差值，便于按时间复查。逐帧数据相互关联，不能把 50 帧简单当作 50 次独立训练重复。','small')
    p.finish()

    p=Page(10,'图像 / QUALITATIVE CHECK','GT 与渲染：同时展示收益和失败')
    p.heading('帧 73：H3 减少 C1 的局部伪影（+4.5212 dB）')
    p.image(A/'crop_73.png',148,'图 3：GT | C1 | H3 | F1。相同裁切框 x=463-843，y=340-670；坐标按原始 1600×900 图像。')
    p.text('C1 在蓝色平台左上边缘外出现明显黄色条状漂浮物；H3 在该处更接近 GT，F1 仍有小范围黄色/灰色残影。H3 的鸭子表面和轮廓也未完全恢复 GT 的清晰程度。','small')
    p.heading('帧 85：H3 仍有形变与颜色拖影（-6.2382 dB）')
    p.image(A/'crop_85.png',148,'图 4：GT | C1 | H3 | F1。相同裁切框 x=731-1111，y=300-630。')
    p.text('H3 中上方鸭子出现拉长和破碎状边缘，中部蓝色平台上混入明显黄色拖影；该帧 C1 更接近 GT。这里的问题不只是 alpha 偏低，还包含位置/形变与外观耦合误差。','small')
    p.text('两个帧分别按 H3 相对 C1 的最大 PSNR 增益/退化选取，框内再按局部 MSE 差最大选区域，属于有意展示的极端正反例，不代表随机样本。图像仅等比例排版与统一裁切，没有增强、锐化或生成式修改。','small')
    p.finish()

    p=Page(11,'验证 / CONVERGENCE','短实验排名不能替代最终验证')
    p.image(A/'validation.png',205,'图 5：30 张验证图的 PSNR 历史，展示 10k-50k；H3 后期保持领先。')
    p.table([['模型','20k val PSNR','50k val PSNR','50k test PSNR'],*[[k,f"{next(r['metrics']['PSNR'] for r in m[k]['history'] if r['iteration']==20000):.4f}",f"{m[k]['validation']['metrics']['PSNR']:.4f}",f"{m[k]['test']['PSNR']:.4f}"] for k in ('C1','H3','F1','F2')]], [65,145,145,152],highlight=2)
    p.text('F1 在 20k 的验证 PSNR 高于 H3，但其覆盖质量护栏未全部通过；到 50k，验证/测试均明显落后 H3。自动根据局部短实验提出修正并不保证最终有效，需要保留完整诊断和固定预算对照。')
    p.text('H3 在验证集的 PSNR 比 C1 高约 0.7118 dB，完整度评分也更低，支持其总体收益不是只出现在测试集。但验证 r50 从 C1 的 0.01447% 增至 H3 的 0.01793%，与测试 r50 的改善方向不同，说明极少数缺口会随时刻变化。')
    p.text('H3 的验证 PSNR 最优快照恰为 50k；验证完整度最优快照为 49k，测试 PSNR 为 40.8614 dB。主表坚持 final50k，没有按测试集结果再挑一个最好迭代。')
    p.finish()

    p=Page(12,'OBJECT ID / ABLATION','ObjectID：模块被使用，但未带来收益')
    p.text('O1/O2 与 C1 共享完整 3k 起点、同一划分和 50k 总预算。它们加入 16 个候选运动槽位，将各 Gaussian 的原形变位置与软归属加权的槽位 SE(3) 位置混合；最大组运动占比 25%，原非刚体路径保留 75%。16 是候选容量，不是真实物体数。')
    p.text('O1 冻结 identity MLP，采用几何软归属；仍用当前 X 查询权重，不是人工指定不变实例标签。O2 学习归属；使用图像重建、轨迹残差、熵和槽位占用代价，不使用实例标签或伪实例目标。两者继承 C1 的训练 alpha，因此不能称为完全不含监督的 RGB-only 系统。')
    rows=[['模型','PSNR','SSIM','LPIPS','MS-SSIM']]
    for k,sc in [('C1',b['test'])]+[(x['id'],x['test']) for x in data['object_id']]:rows.append([k,*[f'{sc[j]:.7f}' for j in ('PSNR','SSIM','LPIPS','MS-SSIM')]])
    p.table(rows,[47,115,115,115,115],highlight=1)
    p.table([['归属诊断','O1','O2'],['平均最大归属概率','0.1722','0.3495'],['pmax≥0.5 的 kernel 比例','0%','14.11%'],['最终活跃候选槽','16 / 16','16 / 16']], [245,131,131])
    p.text('实测：O1/O2 相对 C1 分别下降 1.5613 / 1.8913 dB，四项图像指标均更差；尚未出现足够明确的物体归属。打乱归属会改变渲染，只能证明分支参与了前向计算，不能证明得到正确的 object ID。')
    p.text('可能原因：软归属把不同运动混在一起，组 SE(3) 对真实弹性形变引入偏置，归属与运动同时优化存在歧义。日志支持“当前实现未获益”，但没有实例真值，无法把退化唯一归因于分组错误。')
    p.callout('当前不建议把 O1/O2 叠加到 H3 作为默认配置。后续应先验证分组跨时间稳定性、接触阶段串组和运动残差，再决定是否重新训练。')
    p.finish()

    p=Page(13,'历史 / RELATED EXPERIMENTS','已有其他模块提供了什么线索')
    context={x['name']:x for x in data['context_models']}
    p.table([['比较组（均 final50k）','对照 PSNR','变体 PSNR','可支持的结论'],['颜色修正后的局部仿射',f"{b['test']['PSNR']:.4f}",f"C2 {context['C2_repaired']['test']['PSNR']:.4f}\nC3 {context['C3_repaired']['test']['PSNR']:.4f}",'加入局部仿射及其上的增密未提升该 C1 对照；H2 时间应变不是旧仿射项。'],
        ['固定/可学习共享 E',f"{context['P0_C1_control']['test']['PSNR']:.4f}",f"P1 {context['P1_fixed_E']['test']['PSNR']:.4f}\nP2 {context['P2_learn_E']['test']['PSNR']:.4f}",'各比 P0 低约 2.76-2.77 dB；学习 E 未弥补本构约束导致的退化。'],
        ['原 PIDG + 2D 前景中心',f"{context['G0_control']['test']['PSNR']:.4f}",f"{context['G1_centroid']['test']['PSNR']:.4f}",'PSNR +0.3885 dB，但 SSIM/LPIPS 变差；不是完整 3D 质量中心监督。'],
        ['V3 + iso=0.001',f"{context['V3_scratch_control']['test']['PSNR']:.4f}",f"{context['V3_scratch_iso_1e3']['test']['PSNR']:.4f}",'PSNR -0.1658 dB，LPIPS 变差；SSIM/MS-SSIM 略改善，属于取舍。']], [147,82,111,167])
    p.text('这些对照同样使用 120/30/50 时间划分，但各自有专门的共同起点和训练配方。P1/P2 应与 P0 比较，G1 应与 G0 比较；不可用它们与 H3 的绝对差直接推断某一个模块的因果贡献。P0 的 PSNR 比复用 C1 低约 0.335 dB，也提醒单次运行差异不可忽略。')
    p.text('旧 C1_mask 的颜色退化经“正值辐射 + 初始化修正 + 延迟 mask + 前景颜色监督”整体修正后，从 35.7459 提升至 40.3721 dB；这是修正包效果，不足以证明其中任意单项独自贡献全部提升。','small')
    p.heading('从这些实验得到的共识')
    p.text('更强的约束、更大的参数库、更多点或更接近物理的形式都不自动带来更好渲染。约束需要与实际观测、弹性运动和颜色参数化匹配；先抑制模型“用错误颜色补偿透明度”的机制，再以较弱运动先验辅助覆盖，更符合当前证据。')
    p.finish()

    p=Page(14,'分析 / INTERPRETATION','为什么 H3 更好，下一步该改什么')
    p.table([['已观察到的证据','可能机制与适用边界'],['H1/H2 单独均未提高 PSNR；H3 同时提高四项图像指标和平均完整度。','二维覆盖给出“该填哪里”，弱时间应变限制“如何随时间移动”；两者可能互补。单种子组合结果支持这个假设，不等于证明协同的唯一原因。'],
        ['H3 前景 PSNR 31.9512→32.4961；前景平均 MSE 9.376e-4→8.207e-4。','前景外观本身在改善，不只是大量背景使全图指标变好；区域 PSNR 与全图 PSNR 采用不同分母。'],
        ['H3 背景平均 MSE 4.548e-5→4.486e-5；全图平均 MSE 1.344e-4→1.213e-4。','整体误差也下降；这里的 MSE 由保存的 PNG 重算，与“逐图 PSNR 均值”不是同一聚合形式，不能由 +0.506 dB 直接宣称固定比例的 pooled MSE 下降。'],
        ['H3 87,032 点，仅比 C1 多 889（约 1.03%）；无额外晚期增密。','收益主要伴随训练目标改变出现，而非增加 MLP 容量；但早期常规增密也随优化变化，不能完全排除点分布的贡献。']], [224,283])
    p.heading('建议保留与修改的优先级')
    p.text('<b>默认保留 H3，固定 C1 对照。</b>先用多个随机种子重复 H3/C1，并在独立验证上按时间块而非单帧检查严重失败；不要立即叠加 ObjectID、E 或晚期增密扩大变量数量。')
    p.text('<b>优先处理接触与大形变阶段。</b>帧 85 的形变/颜色拖影提示应检查邻边是否跨物体、运动容差是否足够，以及颜色与 opacity 梯度是否冲突。诊断与改动应先在验证帧上进行；测试帧 85 保留为已暴露案例。')
    p.text('<b>若试修颜色或增密，先做受控实验。</b>F1 可比较只更新颜色与全参数更新；H4 可先可视化新点来源、深度与所属运动，再考虑多时刻累积和增密时序。以上是待检验方案，不是已证实有效的改动。','small')
    p.finish()

    p=Page(15,'归档 / ARTIFACTS','报告、模型与 render 的绝对地址')
    p.heading('七组 GT / render 拼接总览（50 张）')
    gallery=ROOT/'runs/pidg_integrity_continuation/render_exports/final50000/gt_all_models/index.html'
    assert gallery.exists();p.text(path(gallery),'path')
    p.text('总览排列：GT | C1 | H2 | H3 | H1 | H4 | F1 | F2。每个变体另存 renders（纯渲染）及 gt_render（双图拼接）；文件序号 00000-00049 对应原始帧号 1,5,...,197。','small')
    for label,value in [('H3 纯 render',ROOT/'runs/pidg_integrity_continuation/render_exports/final50000/H3_combined/renders'),
        ('H3 与 GT 拼接',ROOT/'runs/pidg_integrity_continuation/render_exports/final50000/H3_combined/gt_render'),
        ('H3 Gaussian 与形变权重目录',Path(h['directory'])),
        ('ObjectID GT/render 总览',ROOT/'runs/pidg_object_id/render_exports/final50000/gt_c1_frozen_learned/index.html')]:
        assert value.exists();p.text('<b>'+label+'</b>','small',gap=2);p.text(path(value),'path',gap=8)
    p.text('H3 权重位于上述模型目录的 point_cloud\\iteration_50000\\point_cloud.ply 与 deform\\iteration_50000\\deform.pth；完整恢复状态为 checkpoints\\resume.pt。','small')
    p.heading('可复核的数字与源码')
    p.text('主要数据：experiments/integrity_continuation/assessment.json；各 run 的 results.json、per_view.json、metric_config.json、metric_history.jsonl、coverage_density.jsonl。ObjectID：experiments/object_id/identity_exports.json。','small')
    p.text('实现：variants/integrity_tuning/{model,runtime}.py；variants/integrity_feedback/{model,runtime}.py；variants/color_repair/{runtime,model,color}.py；variants/object_id/{model,runtime}.py。共享 3k SHA256：','small',gap=2)
    p.text(data['protocol']['shared3k_sha256'],'path',gap=8)
    p.text('本报告旁附同名 Markdown。integrity_analysis_assets/data.json 保存审计数据与来源 SHA256；per_frame_test.csv 保存逐帧 PSNR；crops.json 保存裁切坐标。生成脚本为 scripts/collect_integrity_report_data.py、build_integrity_report_figures.py、build_integrity_technical_report.py。','small')
    p.text('结论范围：当前单场景、单种子已完成实验中的综合最优。没有声称识别真实材质、恢复真实实例、消除所有 3D 空洞，或证明新场景与未来时刻泛化。','small')
    p.finish()
    L.c.save()
    PDF.with_suffix('.md').write_text('# PIDG 完整度与 ObjectID 实验技术分析\n'+'\n\n'.join(RECORD)+'\n',encoding='utf-8')
    manifest={'pdf':str(PDF),'pages':L.page_records,'source_data_sha256':hashlib.sha256((A/'data.json').read_bytes()).hexdigest(),
        'generated':L.REPORT_TIME.isoformat(),'scope':[x['name'] for x in ms]+[x['name'] for x in data['object_id']],
        'context_models':[x['name'] for x in data['context_models']]}
    (A/'report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':build()
