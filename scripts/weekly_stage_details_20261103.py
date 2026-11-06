"""Append audited bilingual S1-S6 configuration details to the frozen reports.

No model imports, training or inference. Original 49-page editions are backed up
once; all original page bodies and embedded images are checked after merging.
"""
from pathlib import Path
from datetime import datetime, timezone
from html import escape
from io import BytesIO
import hashlib
import json
import re
import shutil

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Link
from pypdf.generic import ContentStream, NameObject, ByteStringObject, ArrayObject
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
TMP = ROOT / 'tmp/pdfs/weekly_stage_details_20261103'
W, H = A4
M = 46
CW = W - 2*M
TOTAL = 54
NAVY = colors.HexColor('#16364D')
TEAL = colors.HexColor('#087C83')
INK = colors.HexColor('#1B2B3C')
GRAY = colors.HexColor('#596A78')
LIGHT = colors.HexColor('#EEF4F6')
RULE = colors.HexColor('#D9E3E8')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))

SOURCES = [
    'experiments/staged_reconstruction/S1_protocol.json',
    'experiments/staged_reconstruction/S1_assessment.md',
    'experiments/staged_reconstruction/S2_protocol.json',
    'experiments/staged_reconstruction/S2_assessment.md',
    'experiments/staged_reconstruction/S2_recovery/S2_protocol.json',
    'experiments/staged_reconstruction/S2_recovery/assessment.md',
    'experiments/staged_reconstruction/S2_geometry_recovery/S2_protocol.json',
    'experiments/staged_reconstruction/S2_geometry_recovery/assessment.md',
    'experiments/staged_reconstruction/S3/protocol.json',
    'experiments/staged_reconstruction/S3/assessment.md',
    'experiments/staged_reconstruction/S4/protocol.json',
    'experiments/staged_reconstruction/S4/assessment.md',
    'experiments/staged_reconstruction/S5/protocol.json',
    'experiments/staged_reconstruction/S5/assessment.md',
    'experiments/staged_reconstruction/S5/static_volume_receipt.json',
    'experiments/staged_reconstruction/S6/protocol.json',
    'experiments/staged_reconstruction/S6/assessment.md',
    'experiments/staged_reconstruction/final_summary.md',
    'variants/staged_reconstruction/static_gaussian.py',
    'variants/staged_reconstruction/train_static.py',
    'variants/staged_reconstruction/model.py',
    'variants/staged_reconstruction/stage2/model.py',
    'variants/staged_reconstruction/stage2/train.py',
    'variants/staged_reconstruction/recovery/train.py',
    'variants/staged_reconstruction/recovery/coverage.py',
    'variants/staged_reconstruction/geometry_recovery/model.py',
    'variants/staged_reconstruction/geometry_recovery/train.py',
    'variants/staged_reconstruction/stage3/train.py',
    'variants/staged_reconstruction/stage3/physics.py',
    'variants/staged_reconstruction/state_recovery/physics.py',
    'variants/staged_reconstruction/state_recovery/runtime.py',
    'variants/staged_reconstruction/stage4/constraints.py',
    'variants/staged_reconstruction/stage5/nonpenetration.py',
    'variants/staged_reconstruction/continuation.py',
    'variants/coherent_momentum/operator.py',
    'variants/momentum_guard/balance.py',
]


def pages_for(lang):
    def v(cn, en): return cn if lang == 'CN' else en
    def p(cn, en, kind='body'): return ('p', v(cn, en), kind)
    def h(cn, en): return p(cn, en, 'heading')
    def box(cn, en): return ('box', v(cn, en))
    def table(cn_head, en_head, cn_rows, en_rows, widths):
        return ('table', v(cn_head, en_head), v(cn_rows, en_rows), widths)

    pages = []
    pages.append((v('S1-S6：每一阶段具体改变了什么', 'S1-S6: What changes at each stage'), [
        box('这是一条顺序继承的训练链：图像初始化 → S1 静态 → S2 动态 → S3 物理场 → S4 稀疏轨迹 → S5 条件内部先验 → S6 联合微调。',
            'This is a sequential training curriculum: image initialization > S1 static > S2 dynamic > S3 physics fields > S4 sparse tracks > S5 conditional interior prior > S6 joint fine-tuning.'),
        table(['阶段 / 本阶段步数', '继承与新增内容', '训练参数 / 冻结范围'],
              ['Stage / updates', 'Parent and main change', 'Trainable / frozen parameters'], [
            ['S1 / 12k', '从训练图像初始化；只拟合静态点库', '静态位置、SH颜色、透明度、尺度、旋转可训；无动态点'],
            ['S2 / 16k', '继承S1；加入四组动态点和各组PIDG形变网络', '动态canonical坐标及外观/几何、形变网格/网络可训；静态全冻结'],
            ['S3 / +6k', '继承正式S2；新增每组速度/应力场和运动学、CMR目标', '保留S2可训练参数，新增物理场参数；静态全冻结'],
            ['S4 / +4k', '继承S3；新增固定Gaussian行的2D轨迹监督', '不加新点参数；轨迹回传到canonical坐标和形变；静态全冻结'],
            ['S5 / +4k', '继承S4；启用条件静态深内部非穿透先验', '参数策略同S4；固定条件SDF不训练'],
            ['S6 / +4k', '继承S5；各组学习率×0.25，解冻静态SH颜色', '动态与物理参数继续训练；静态位置/尺度/旋转/透明度仍冻结'],
        ], [
            ['S1 / 12k', 'Initialize from training images; fit a static Gaussian bank', 'Train static positions, SH color, opacity, scale and rotation; no dynamic bank'],
            ['S2 / 16k', 'Keep S1; add four dynamic banks and independent PIDG deformation models', 'Train dynamic canonical positions, appearance/geometry, deformation grids and networks; freeze all static tensors'],
            ['S3 / +6k', 'Continue accepted S2; add per-bank velocity/stress fields, kinematics and CMR', 'Keep S2 trainables; add physics-field parameters; freeze static bank'],
            ['S4 / +4k', 'Continue S3; add 2D tracks bound to fixed Gaussian rows', 'No new point parameters; tracks update canonical positions and deformation; freeze static bank'],
            ['S5 / +4k', 'Continue S4; enable a conditional static deep-interior prior', 'Same trainables as S4; conditional SDF remains fixed'],
            ['S6 / +4k', 'Continue S5; multiply all learning rates by 0.25 and unfreeze static SH color', 'Keep dynamic/physics training; static position, scale, rotation and opacity stay frozen'],
        ], [.85, 1.65, 2.1]),
        h('如何理解“固定”与模型编号', 'Meaning of fixed banks and stage labels'),
        p('固定的是动态组成员、拓扑和轨迹绑定行号；dynamic canonical坐标 _xyz 仍在优化器中，并非空间位置固定。四组是结构模型分组，尚不是经验证的真实物体身份。',
          'Bank membership, topology and bound row IDs are fixed. Dynamic canonical coordinates (_xyz) remain in the optimizer and can move. The four banks are structural model groups, with no verified persistent physical identities.'),
        p('正式继承链合计46k更新；另外两次失败S2各16k，共32k，单独计入尝试预算。这里的Stage S1不是历史S1_scratch_constraints，历史S0也不是Stage S1的父模型。',
          'The accepted chain totals 46k updates. Two failed S2 attempts add 32k updates separately. Stage S1 is distinct from historical S1_scratch_constraints; historical S0 is not its parent.'),
        p('S1/S2分辨率逐步提升，S3-S6均在1600×900追加训练。各阶段继承前态，新增目标和训练时长同时变化，因此阶段间差值不能直接解释为单项损失的因果贡献。',
          'S1/S2 increase image resolution during training; S3-S6 continue at 1600 x 900. Since objectives and training duration change together, stage differences do not isolate the causal contribution of one loss.'),
        p('本增补核对正式protocol、训练代码和assessment；不改变p1-49的实验数字、图像和test逐帧数据。阶段训练不等于PIDG扩展M0-M10全部完成。',
          'This addendum audits formal protocols, implementation and assessments. Experimental values, images and per-frame test data on pp. 1-49 are preserved. These stages do not establish completion of extension milestones M0-M10.', 'small'),
    ]))
    pages.append((v('S1 / S2：静态点库与动态重建', 'S1 / S2: Static bank and dynamic reconstruction'), [
        h('S1：从图像独立建立静态基底', 'S1: Build the static bank from images'),
        p('初始化来自训练图像跨时刻轮廓视锥约束，使用metadata CV世界坐标，没有历史训练检查点。实际4,126个静态Gaussian，不增密。可训练 _xyz、_features_dc、_features_rest、_opacity、_scaling、_rotation。',
          'Initialize from training-only multitime silhouette constraints in metadata CV world coordinates, with no trained checkpoint. The bank contains 4,126 Gaussians with no densification. Train _xyz, _features_dc, _features_rest, _opacity, _scaling and _rotation.'),
        p('蓝色静态候选提供RGB/前景alpha正监督，确定背景提供alpha负监督；黄色遮挡及其他未知前景不作为黑色目标。目标如下，均在对应掩码中取均值：',
          'Blue static candidates supervise RGB and foreground alpha; certain background supervises low alpha. Yellow occlusion and other unknown foreground are excluded. The objective uses means on the respective masks:'),
        box('L_S1 = L1_RGB(positive) + 0.2 MSE(alpha_positive, 1) + 0.2 MSE(alpha_background, 0) + 0.01 mean((X - X_init)^2)',
            'L_S1 = L1_RGB(positive) + 0.2 MSE(alpha_positive, 1) + 0.2 MSE(alpha_background, 0) + 0.01 mean((X - X_init)^2)'),
        p('训练12k步：前8k宽800，之后宽1600；正式评估1600×900。S1无动态、CMR、接触或轨迹损失。静态候选PSNR：val 35.64382 / test 35.79119 dB。',
          'Train 12k updates: width 800 for the first 8k, then 1600; formal evaluation is 1600 x 900. No dynamic, CMR, contact or track objective. Static-candidate PSNR: val 35.64382 / test 35.79119 dB.'),
        h('S2：正式采用第三次尝试 S2_geometry_recovery', 'S2: Use the third attempt, S2_geometry_recovery'),
        p('冻结全部S1静态张量；加入15,507个动态点、四组局部canonical点库和四个独立原PIDG DeformModel。动态六类Gaussian参数及形变网格/网络可训，点数与成员不变。形变网络depth=1、width=128；canonical grid 12层/hash15/res256，deform grid 16层/hash15/res[32,32,16]。',
          'Freeze every S1 static tensor. Add 15,507 dynamic points in four local canonical banks, each with an original PIDG DeformModel. Train all six dynamic Gaussian parameter types plus deformation grids/networks; keep count and membership fixed. Network depth=1, width=128; canonical grid: 12 levels/hash15/res256; deformation grid: 16 levels/hash15/res[32,32,16].'),
        box('L_img = 0.8 L1_RGB(all) + 0.2 (1 - SSIM) + 0.5 L1_RGB(yellow) + 2.0 L_coverage + 0.5 L1_RGB(blue)',
            'L_img = 0.8 L1_RGB(all) + 0.2 (1 - SSIM) + 0.5 L1_RGB(yellow) + 2.0 L_coverage + 0.5 L1_RGB(blue)'),
        p('RGB目标为alpha预乘图像。coverage是观测前景像素到投影动态点的单侧最近平方距离，坐标除以图宽；已合格的早期组用分组覆盖，歧义组用无标签并集，不把隐藏点强压进可见掩码。动态尺度限制在[0.15, 2.5]×canonical spacing；深度薄层是初始化假设，非深度GT。',
          'RGB uses alpha-premultiplied targets. Coverage measures one-way nearest squared distance from observed foreground pixels to projected dynamic points, normalized by image width. Qualified early groups use labeled coverage; ambiguous groups use an unlabeled union. Hidden points are not forced into the visible mask. Bound dynamic scales to [0.15, 2.5] times canonical spacing. The initial depth slab is a hypothesis, not depth GT.'),
        p('首版S2的val动态/静态PSNR为9.3514/33.8937；第一次修正为25.0864/31.3350，静态退步且尺度退化。正式版加入尺度约束和对称静态RGB后为24.25343/35.53011 dB。三次各16k；正式版前12k宽800，之后1600。仍无物理/轨迹，固定视角依旧碎裂。',
          'Initial S2 val dynamic/static PSNR: 9.3514/33.8937 dB. First repair: 25.0864/31.3350, with static damage and scale degeneration. The accepted repair adds bounded scales and symmetric static RGB, reaching 24.25343/35.53011 dB. Each attempt uses 16k updates; accepted S2 uses width 800 for 12k, then 1600. Physics/tracks remain off; fixed views still fragment.'),
        p('来源：S1_protocol.json；S2_geometry_recovery/S2_protocol.json；static_gaussian.py、train_static.py、stage2/model.py、recovery/coverage.py、geometry_recovery/{model,train}.py。完整路径与SHA见.sources.stages.json。',
          'Sources: S1_protocol.json; S2_geometry_recovery/S2_protocol.json; static_gaussian.py, train_static.py, stage2/model.py, recovery/coverage.py and geometry_recovery/{model,train}.py. Full paths and hashes are in .sources.stages.json.', 'source'),
    ]))
    pages.append((v('S3：接入速度、应力和动量约束', 'S3: Add velocity, stress and momentum constraints'), [
        h('相对正式S2增加什么', 'What is added to accepted S2'),
        p('从正式S2终态恢复模型和优化器，追加6000步。S2的L_img、动态可训练参数和静态全冻结策略保持；每组新增一个原PIDG VelocityNet，输出速度v和预测应力sigma_pred，关闭自由逐点index embedding。新增物理场Adam学习率0.001。',
          'Restore the accepted S2 model and optimizer, then train 6,000 more updates. Keep L_img, all dynamic trainables and the fully frozen static bank. Add an original PIDG VelocityNet per bank to predict velocity v and stress sigma_pred, with free per-point index embeddings disabled. New physics parameters use Adam learning rate 0.001.'),
        box('L_S3 = L_img + 0.01 L_kinematic + 0.00001 L_CMR\nv_traj(X,t) = [x(X,t+h) - x(X,t-h)] / (2h)\nL_kinematic = Huber(v(X,t), v_traj(X,t))',
            'L_S3 = L_img + 0.01 L_kinematic + 0.00001 L_CMR\nv_traj(X,t) = [x(X,t+h) - x(X,t-h)] / (2h)\nL_kinematic = Huber(v(X,t), v_traj(X,t))'),
        p('h=1/199；边界使用夹到序列内部的中心时刻及其两侧差分。物质加速度是在固定canonical坐标X对预测速度做时间差分。rho固定1，rate coefficient固定1，应力散度保留预测应力和PIDG速度率项：',
          'Use h=1/199. At sequence boundaries, clamp the stencil center into the interior. Material acceleration is a time difference of predicted velocity at fixed canonical X. Density rho=1 and rate coefficient=1 are fixed. The stress divergence retains predicted stress and the PIDG velocity-rate term:'),
        box('a_mat = [v(X,t+h) - v(X,t-h)] / (2h)\nr_CMR = a_mat - div_x(sigma_pred) - [0.5 Laplacian(v) + grad(div(v))/6]\nL_CMR = Huber(r_CMR, 0) on qualified full-rank neighborhoods',
            'a_mat = [v(X,t+h) - v(X,t-h)] / (2h)\nr_CMR = a_mat - div_x(sigma_pred) - [0.5 Laplacian(v) + grad(div(v))/6]\nL_CMR = Huber(r_CMR, 0) on qualified full-rank neighborhoods'),
        h('真正执行的空间离散与梯度策略', 'Executed spatial discretization and gradient policy'),
        p('正式恢复版本在当前世界坐标内建同组邻域，用二次加权最小二乘估计应力一阶导数和速度二阶导数；依次尝试64/128/256邻点，拒绝退化支持，每组至少25%锚点合格。最初canonical邻域恢复短训失败，被修正后才完成正式6k。',
          'The accepted recovery builds same-bank neighborhoods in current world coordinates. Quadratic weighted least squares estimate stress first derivatives and velocity second derivatives. Try 64/128/256 neighbors and reject deficient supports; at least 25% of anchors per bank must qualify. An initial recovery smoke run failed on canonical neighborhoods; the full 6k run follows this correction.'),
        p('物理目标每10步计算，每组16锚点。形变网格、网络和canonical坐标共享梯度受0.1比例保护，物理私有参数梯度保留。这个梯度策略不等价于全程简单相加一个固定权重标量损失；上式表示原始目标。',
          'Evaluate physics every 10 updates, with 16 anchors per bank. Shared gradients into deformation grids, networks and canonical coordinates are guarded at ratio 0.1 relative to image gradients; private physics gradients are preserved. Thus optimization is not simply an unmodified fixed weighted scalar sum; the equations show the raw objectives.'),
        h('解释范围', 'Interpretation'),
        p('val动态PSNR为25.392299 dB；仍有固定视角薄片、空洞和组泄漏。这里没有新增本构关系、可学习刚度、接触、稠密LPFM、外力或物理身份监督。时间是序列归一化单位，真实采样秒数未知；内部残差下降不能证明速度/应力符合真实物理。',
          'Val dynamic PSNR is 25.392299 dB; fixed views still show sheets, holes and bank leakage. No constitutive law, learnable stiffness, contact, dense LPFM, external force or physical-identity supervision is added. Time is sequence-normalized; acquisition seconds are unknown. Internal residual reduction does not validate physical velocity or stress.'),
        p('来源：S3/protocol.json与assessment.md；stage3/{train,physics}.py；state_recovery/{physics,runtime}.py；coherent_momentum/operator.py；momentum_guard/balance.py。',
          'Sources: S3/protocol.json and assessment.md; stage3/{train,physics}.py; state_recovery/{physics,runtime}.py; coherent_momentum/operator.py; momentum_guard/balance.py.', 'source'),
    ]))
    pages.append((v('S4 / S5：轨迹绑定与条件内部先验', 'S4 / S5: Bound tracks and conditional interior prior'), [
        h('S4：用训练图像轨迹约束同一Gaussian行', 'S4: Constrain fixed Gaussian rows with image tracks'),
        p('继承S3追加4000步；保留L_img、运动学和CMR。新增91条合格短轨迹、455次图像观测，绑定79个固定Gaussian行。每5步抽4条轨迹，不增加可学习匹配器或新点参数。绑定后不重新分配行号；梯度进入动态canonical坐标和形变网格/网络。静态全部冻结。',
          'Continue S3 for 4,000 updates, retaining L_img, kinematics and CMR. Add 91 qualified tracklets comprising 455 image observations bound to 79 fixed Gaussian rows. Sample four tracks every five updates. No learned matcher or new point parameters are added. Keep bindings fixed; gradients update dynamic canonical coordinates and deformation grids/networks. Freeze the entire static bank.'),
        box('L_S4 = L_S3 + 0.1 L_track\nL_track: confidence-weighted Huber(project(x_row(t)), uv_track), delta=1 px, normalized by width=400',
            'L_S4 = L_S3 + 0.1 L_track\nL_track: confidence-weighted Huber(project(x_row(t)), uv_track), delta=1 px, normalized by width=400'),
        p('每次观测对u/v取Huber均值；当前调用一次只传一个合格观测，其正confidence在归一化时相消，最终各观测等权平均。共享轨迹梯度也有0.1保护。全455次观测误差从S3的0.536841升到S4的0.785017 px@400，反而变差。轨迹来自训练图像，并非独立3D、长遮挡重识别或物理身份GT。',
          'Average Huber over u/v for each observation. Current calls each contain one qualified observation, so positive confidence cancels during normalization; observations are then averaged equally. Shared track gradients use the 0.1 guard. Mean error over all 455 observations worsens from S3 0.536841 to S4 0.785017 px@400. These are training-image correspondences, not independent 3D, long-occlusion re-identification or physical-identity GT.'),
        h('S5：仅对合格观测点检查条件静态深内部', 'S5: Check only qualified points against conditional static interior'),
        p('继承S4追加4000步，学习率倍率仍1.0。保持S4的绑定、目标和冻结策略，额外启用固定SDF的非穿透先验。仅在独立图像合格的固定行、其实际观测时刻上评估，与轨迹同为每5步；不对所有隐藏点施加接触假设。',
          'Continue S4 for 4,000 updates, with learning-rate multiplier 1.0. Keep S4 bindings, objectives and freezing policy; enable an extra prior using a fixed conditional SDF. Evaluate only independently image-qualified rows at their observed times, every five updates alongside tracks. No contact assumption is imposed on all hidden points.'),
        box('L_S5 = L_S4 + 0.01 L_interior\nL_interior = mean(max(-d_conditional(x) - 0.08, 0)^2)',
            'L_S5 = L_S4 + 0.01 L_interior\nL_interior = mean(max(-d_conditional(x) - 0.08, 0)^2)'),
        p('d为条件重建静态体的有符号距离；margin=0.08场景单位，只有进入负距离深处超过margin才激活。SDF不训练；若激活，位置梯度可回传到动态canonical和形变。该项与track合并后受共享梯度保护。',
          'd is the signed distance to a conditionally reconstructed static volume. The margin is 0.08 scene units, so activation requires penetration deeper than this margin. The SDF is fixed. If active, position gradients can reach dynamic canonical coordinates and deformation. The combined track/interior objective uses the shared gradient guard.'),
        box('S5实际：800次调用、16,000个合格观测实例，激活违规=0，非零先验=0。因此没有发生真实训练中的碰撞修正。受控球/平面300步仅验证算子。',
            'Observed in S5: 800 calls, 16,000 qualified observation instances, zero active violations and zero nonzero prior calls. No collision correction occurred in real-scene training. The 300-step sphere/plane fixture validates only the operator.'),
        p('S5 val动态PSNR 25.603743 dB，轨迹误差0.66467054 px@400，仍高于S3。S4/S5均未加入物体间接触力、摩擦、恢复系数、本构或未知外力。',
          'S5 val dynamic PSNR is 25.603743 dB; track error is 0.66467054 px@400, still above S3. Neither S4 nor S5 adds object-object contact forces, friction, restitution, constitutive laws or unknown external forces.'),
        p('来源：S4/S5 protocol.json与assessment.md；stage4/constraints.py；stage5/nonpenetration.py；S5/static_volume_receipt.json；continuation.py。',
          'Sources: S4/S5 protocol.json and assessment.md; stage4/constraints.py; stage5/nonpenetration.py; S5/static_volume_receipt.json; continuation.py.', 'source'),
    ]))
    pages.append((v('S6：联合微调及阶段开关总表', 'S6: Joint fine-tuning and objective switches'), [
        h('相对S5只有两类训练策略变化', 'Two optimization changes relative to S5'),
        p('继承S5模型、优化器和采样状态，再训练4000步，所有已有目标继续启用。各优化器参数组学习率统一乘0.25；只解冻静态 _features_dc 和 _features_rest，即SH颜色。没有解冻静态透明度 _opacity，静态位置 _xyz、尺度 _scaling、旋转 _rotation 也保持S1。新静态颜色组基础lr=0.000125，乘0.25后为0.00003125。',
          'Restore the S5 model, optimizer and sampling state for 4,000 more updates, keeping every existing objective. Multiply all optimizer-group learning rates by 0.25. Unfreeze only static _features_dc and _features_rest (SH color). Static _opacity, _xyz, _scaling and _rotation remain fixed at S1. The new color group starts at 0.000125 before scaling, giving 0.00003125.'),
        box('L_S6 = L_img + 0.01 L_kinematic + 0.00001 L_CMR + 0.1 L_track + 0.01 L_interior\nPhysics: every 10 steps / 16 anchors per bank. Tracks + interior: every 5 steps / 4 tracks. Shared gradient cap: 0.1.',
            'L_S6 = L_img + 0.01 L_kinematic + 0.00001 L_CMR + 0.1 L_track + 0.01 L_interior\nPhysics: every 10 steps / 16 anchors per bank. Tracks + interior: every 5 steps / 4 tracks. Shared gradient cap: 0.1.'),
        table(['目标 / 策略', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
              ['Objective / policy', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6'], [
            ['S1专用掩码目标', '开', '-', '-', '-', '-', '-'],
            ['平衡图像目标 L_img', '-', '开', '开', '开', '开', '开'],
            ['运动学 / CMR 权重', '-', '-', '.01 / 1e-5', '.01 / 1e-5', '.01 / 1e-5', '.01 / 1e-5'],
            ['固定行轨迹权重', '-', '-', '-', '.1', '.1', '.1'],
            ['条件内部先验权重', '-', '-', '-', '-', '.01*', '.01*'],
            ['静态SH颜色可训练', '是', '否', '否', '否', '否', '是'],
            ['静态几何/透明度可训练', '是', '否', '否', '否', '否', '否'],
            ['本阶段更新 / lr倍率', '12k / 1', '16k / 1', '6k / 1', '4k / 1', '4k / 1', '4k / .25'],
        ], [
            ['S1 masked objective', 'On', '-', '-', '-', '-', '-'],
            ['Balanced image L_img', '-', 'On', 'On', 'On', 'On', 'On'],
            ['Kinematic / CMR weight', '-', '-', '.01 / 1e-5', '.01 / 1e-5', '.01 / 1e-5', '.01 / 1e-5'],
            ['Fixed-row track weight', '-', '-', '-', '.1', '.1', '.1'],
            ['Conditional interior weight', '-', '-', '-', '-', '.01*', '.01*'],
            ['Train static SH color', 'Yes', 'No', 'No', 'No', 'No', 'Yes'],
            ['Train static geometry/opacity', 'Yes', 'No', 'No', 'No', 'No', 'No'],
            ['Updates / LR multiplier', '12k / 1', '16k / 1', '6k / 1', '4k / 1', '4k / 1', '4k / .25'],
        ], [2.1, .7, .7, .85, .85, .85, .85]),
        p('* S5/S6先验代码已启用，但每阶段全程800次调用均为零激活、零非零先验；不能将其记为实际有效的碰撞学习。S1使用独立静态目标；S2-S6的图像目标定义见p51。阶段损失按各自频率执行，且有p52所述梯度保护。',
          '* S5/S6 enable the prior in code, but all 800 calls per stage have zero activation and zero nonzero prior values. This does not establish collision learning. S1 uses its own static objective; pp. 51-52 define L_img, objective schedules and gradient guards.', 'small'),
        h('S6提升该如何归因', 'How to interpret the S6 improvement'),
        p('S6 val动态PSNR 28.327031 dB；轨迹0.42710676 px@400，相对S3下降20.44%。静态几何精确保留S1，静态SH颜色有更新。提升来自顺序训练、较低学习率、颜色解冻及既有目标共同作用，现有实验不能分离各项贡献。固定视角完整几何、遮挡后真实身份、接触与材料参数仍未验证。',
          'S6 val dynamic PSNR is 28.327031 dB; track error is 0.42710676 px@400, 20.44% below S3. Static geometry remains exactly S1 while static SH color changes. Sequential training, lower learning rates, color unfreezing and existing objectives act together; these experiments cannot separate their contributions. Complete fixed-view geometry, identity after occlusion, contact and material parameters remain unvalidated.'),
        p('来源：S6/protocol.json、assessment.md和continuation.py。图像结果见p31-34；H3与S0-S6的共同test逐帧比较见p35-47。本增补双语源码与全部来源SHA记录于scripts/weekly_stage_details_20261103.py及.sources.stages.json。',
          'Sources: S6/protocol.json, assessment.md and continuation.py. Image results: pp. 31-34. Common-test per-frame comparison of H3 and S0-S6: pp. 35-47. Bilingual addendum source and source hashes: scripts/weekly_stage_details_20261103.py and .sources.stages.json.', 'source'),
    ]))
    return pages


def register_fonts():
    for name, file in [('CN', 'Deng.ttf'), ('CNB', 'Dengb.ttf'), ('EN', 'arial.ttf'), ('ENB', 'arialbd.ttf')]:
        pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/' + file))
    for name in ('CN', 'EN'):
        pdfmetrics.registerFontFamily(name, normal=name, bold=name+'B', italic=name, boldItalic=name+'B')


def style(lang, kind='body', scale=1):
    settings = {
        'body': (9.4, 13.4, INK, lang),
        'heading': (11.5, 15.1, TEAL, lang+'B'),
        'small': (8.2, 11.2, GRAY, lang),
        'source': (7.6, 10.3, GRAY, lang),
        'cell': (8.2, 11.0, INK, lang),
        'head': (8.0, 10.6, colors.white, lang+'B'),
        'box': (9.5, 13.6, NAVY, lang+'B'),
    }
    size, leading, color, font = settings[kind]
    return ParagraphStyle(kind, fontName=font, fontSize=size*scale, leading=leading*scale,
                          textColor=color, wordWrap='CJK' if lang=='CN' else None,
                          rightIndent=size*scale if lang=='CN' else 0,
                          splitLongWords=True)


def paragraph(text, lang, kind='body', scale=1):
    return Paragraph(escape(text).replace('\n', '<br/>'), style(lang, kind, scale))


def plan(blocks, lang, scale=1):
    items = []
    for block in blocks:
        if block[0] == 'p':
            _, text, kind = block
            obj = paragraph(text, lang, kind, scale)
            _, height = obj.wrap(CW, H)
            items.append((obj, height, 8*scale if kind!='heading' else 7*scale, False))
        elif block[0] == 'box':
            obj = paragraph(block[1], lang, 'box', scale)
            _, height = obj.wrap(CW-28, H)
            items.append((obj, height+22*scale, 11*scale, True))
        else:
            _, headers, rows, weights = block
            data = [[paragraph(s, lang, 'head', scale) for s in headers]]
            data += [[paragraph(s, lang, 'cell', scale) for s in row] for row in rows]
            obj = Table(data, colWidths=[CW*x/sum(weights) for x in weights])
            commands = [('BACKGROUND', (0,0), (-1,0), NAVY),
                        ('VALIGN', (0,0), (-1,-1), 'TOP'),
                        ('TOPPADDING', (0,0), (-1,-1), 5*scale),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 5*scale),
                        ('LEFTPADDING', (0,0), (-1,-1), 5*scale),
                        ('RIGHTPADDING', (0,0), (-1,-1), 5*scale),
                        ('LINEBELOW', (0,1), (-1,-1), .3, RULE)]
            for row in range(2, len(data), 2): commands.append(('BACKGROUND',(0,row),(-1,row),LIGHT))
            obj.setStyle(TableStyle(commands))
            _, height = obj.wrap(CW, H)
            items.append((obj, height, 10*scale, False))
    return items, sum(height+gap for _,height,gap,_ in items)


def footer(c, lang, number):
    c.setStrokeColor(RULE); c.setLineWidth(.5); c.line(M,37,W-M,37)
    c.setFillColor(GRAY); c.setFont(lang,7.4)
    c.drawString(M,24,'2026.10.26 - 2026.11.03 | Rubber Duck')
    c.drawRightString(W-M,24,f'{number:02d} / {TOTAL}')


def build_addendum(lang):
    path = TMP / f'addendum_{lang}.pdf'
    c = canvas.Canvas(str(path), pagesize=A4)
    layout=[]
    for number,(title,blocks) in enumerate(pages_for(lang),50):
        for scale in (1., .98, .96, .94, .92, .90, .88):
            items, height = plan(blocks,lang,scale)
            if height <= H-154: break
        else: raise ValueError(('Overflow',lang,number,height))
        c.setFillColor(TEAL); c.rect(M,H-38,26,3,fill=1,stroke=0)
        c.setFillColor(GRAY); c.setFont(lang,7.8)
        c.drawString(M+34,H-38,'APPENDIX / STAGE CONFIGURATION AUDIT')
        c.drawRightString(W-M,H-38,'PIDG / S1-S6')
        ts=min(20.5,CW/pdfmetrics.stringWidth(title,lang+'B',1))
        c.setFont(lang+'B',ts); c.setFillColor(NAVY); c.drawString(M,H-79,title)
        y=H-101
        for obj,height,gap,is_box in items:
            if is_box:
                c.setFillColor(LIGHT); c.roundRect(M,y-height,CW,height,5,fill=1,stroke=0)
                c.setFillColor(TEAL); c.rect(M,y-height,3,height,fill=1,stroke=0)
                obj.drawOn(c,M+14,y-height+11*scale)
            else: obj.drawOn(c,M,y-height)
            y-=height+gap
        assert y>=49,(lang,number,y)
        footer(c,lang,number)
        c.showPage()
        layout.append({'page':number,'scale':scale,'bottom_content_y':round(y,2)})
    c.save()
    return path,layout


def remove_old_page_total(page, reader):
    content=ContentStream(page.get_contents(),reader)
    count=0
    ops=[]
    for operands,operator in content.operations:
        text=''
        if operator == b'Tj': text=str(operands[0])
        elif operator == b'TJ': text=''.join(str(x) for x in operands[0] if isinstance(x,str))
        if re.fullmatch(r'\d{2} / 49',text): count+=1; continue
        ops.append((operands,operator))
    assert count==1,('Original footer not uniquely found',count)
    content.operations=ops
    page[NameObject('/Contents')]=content


def repair_english_punctuation(page, reader):
    """Replace legacy .notdef punctuation with a visible ASCII comma."""
    content=ContentStream(page.get_contents(),reader)
    count=0
    for operands,operator in content.operations:
        strings=[(operands,0)] if operator==b'Tj' else (
            [(operands[0],i) for i,x in enumerate(operands[0]) if isinstance(x,(str,bytes))]
            if operator==b'TJ' else [])
        for values,index in strings:
            obj=values[index]
            raw=obj.original_bytes if hasattr(obj,'original_bytes') else bytes(obj)
            if b'\x00' in raw:
                count+=raw.count(b'\x00')
                values[index]=ByteStringObject(raw.replace(b'\x00',b','))
    page[NameObject('/Contents')]=content
    return count


def overlay(lang, number):
    stream=BytesIO()
    c=canvas.Canvas(stream,pagesize=A4)
    c.setFont(lang,7.4); c.setFillColor(GRAY)
    c.drawRightString(W-M,24,f'{number:02d} / {TOTAL}')
    if number in (1,14):
        if number==1:
            title='新增：各阶段改动详解（p50-54）' if lang=='CN' else 'Added: Stage-by-stage changes (pp. 50-54)'
            text=('逐项列出S1-S6的继承关系、可训练/冻结参数、损失权重及实际激活情况。' if lang=='CN' else
                  'S1-S6 parent states, trainable/frozen parameters, loss weights and observed activation.')
            y=65; height=57
        else:
            title='阶段配置详解见p50-54' if lang=='CN' else 'Detailed stage configurations: pp. 50-54'
            text=('“固定canonical组”指成员与行号固定；canonical坐标仍可训练。S6只解冻静态SH颜色，静态透明度与几何仍冻结。新增目标、权重及训练预算见后附五页。' if lang=='CN' else
                  'Fixed canonical banks refer to membership and row IDs; canonical positions remain trainable. S6 unfreezes static SH color only; static opacity and geometry stay frozen. The five-page addendum lists objectives, weights and training budgets.')
            y=105; height=82
        c.setFillColor(LIGHT); c.roundRect(M,y,CW,height,5,fill=1,stroke=0)
        obj=paragraph(title,lang,'heading'); _,th=obj.wrap(CW-24,H)
        obj.drawOn(c,M+12,y+height-10-th)
        obj=paragraph(text,lang,'small'); _,bh=obj.wrap(CW-24,H)
        assert th+bh+24<=height
        obj.drawOn(c,M+12,y+height-15-th-bh)
    c.showPage(); c.save(); stream.seek(0)
    return PdfReader(stream).pages[0]


def image_fingerprints(reader):
    return [[hashlib.sha256(image.data).hexdigest() for image in page.images] for page in reader.pages]


def verify_body_preserved(original, updated, lang):
    for i,page in enumerate(original.pages):
        before=page.extract_text()
        if lang=='EN': before=before.replace('\x00',',')
        after=updated.pages[i].extract_text()
        before=re.sub(r'\b\d{2} / 49\b','',before)
        cursor=0
        # Preserve the original extracted sequence, ignoring only the replaced footer.
        for line in before.splitlines():
            if not line.strip():continue
            index=after.find(line,cursor)
            assert index>=0,('Original text changed',lang,i+1,line)
            cursor=index+len(line)
    assert image_fingerprints(original)==image_fingerprints(type('Pages',(),{'pages':updated.pages[:49]})())


def main():
    TMP.mkdir(parents=True,exist_ok=True)
    register_fonts()
    source_manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
                     'sources':{},'training_or_inference':False}
    for name in SOURCES:
        path=ROOT/name
        assert path.is_file(),name
        destination=TMP/'evidence'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
        source_manifest['sources'][name]={'sha256':sha(path),'snapshot':str(destination.relative_to(ROOT))}
    for lang in ('CN','EN'):
        base=f'PIDG_Weekly_Experiment_Summary_{lang}_20261103'
        pdf=OUT/(base+'.pdf')
        backup=TMP/(base+'.original49.pdf')
        if not backup.exists():
            assert len(PdfReader(pdf).pages)==49
            shutil.copyfile(pdf,backup)
            for suffix in ('.qa.json','.source.txt'):
                shutil.copyfile(OUT/(base+suffix),TMP/(base+'.original49'+suffix))
        original=PdfReader(backup)
        # Navigation must occupy genuine whitespace, not merely preserve hidden text.
        with pdfplumber.open(backup) as doc:
            for index,bottom,top in ((0,65,122),(13,105,187)):
                covered=[x['text'] for x in doc.pages[index].chars if x['text'].strip()
                         and x['x1']>M and x['x0']<W-M
                         and x['top']<H-bottom and x['bottom']>H-top]
                assert not covered,('Navigation overlaps original text',lang,index+1,covered)
        addendum,layout=build_addendum(lang)
        writer=PdfWriter(clone_from=backup)
        repaired=0
        for i,page in enumerate(writer.pages):
            remove_old_page_total(page,writer)
            if lang=='EN':repaired+=repair_english_punctuation(page,writer)
            page.merge_page(overlay(lang,i+1))
        assert repaired==(5 if lang=='EN' else 0)
        writer.append(PdfReader(addendum),import_outline=False)
        writer.add_outline_item('S1-S6: Stage configuration details',49)
        for i,(title,_) in enumerate(pages_for(lang)):
            writer.add_outline_item(title,49+i)
        for i,rect in ((0,(M,65,W-M,122)),(13,(M,105,W-M,187))):
            link=writer.add_annotation(page_number=i,annotation=Link(rect=rect,target_page_index=49))
            link[NameObject('/Dest')]=ArrayObject([writer.pages[49].indirect_reference,NameObject('/Fit')])
        writer.add_metadata({'/Subject':'Experiment summary with audited S1-S6 stage configuration details, pp. 50-54'})
        staged=TMP/(base+'.updated54.pdf')
        with staged.open('wb') as stream: writer.write(stream)
        updated=PdfReader(staged)
        assert len(updated.pages)==TOTAL
        verify_body_preserved(original,updated,lang)
        new_text='\n'.join(page.extract_text() for page in updated.pages[49:])
        for token in ('0.00003125','0.785017','0.42710676','0.08','0.25','15,507','4,126'):
            assert token in new_text,(lang,token)
        if lang=='EN':assert not re.search('[\u3400-\u9fff]',new_text)
        assert not any('\x00' in page.extract_text() for page in updated.pages)
        bounds=[]
        with pdfplumber.open(staged) as doc:
            for i in list(range(49,54))+[0,13]:
                page=doc.pages[i]
                bad=[(x['text'],x['x0'],x['top']) for x in page.chars
                     if x['text'].strip() and (x['x0']<M-2 or x['x1']>W-M+2 or x['top']<20 or x['bottom']>H-12)]
                assert not bad,(lang,i+1,bad[:10])
                bounds.append(i+1)
        qa=read(TMP/(base+'.original49.qa.json'))
        qa['original49_generator_sha256']=qa.get('generator_sha256')
        qa['generator_sha256']=sha(__file__)
        if isinstance(qa.get('layout'),list):qa['layout']=qa['layout']+layout
        if lang=='EN':
            qa['original49_source_CN_pdf_sha256']=qa.get('source_CN_pdf_sha256')
            qa['source_CN_pdf_sha256']=sha(OUT/'PIDG_Weekly_Experiment_Summary_CN_20261103.pdf')
            qa['source_CN_artifacts_unchanged']=False
            qa['bilingual_reports_updated_together']=True
        qa.update({'pages':TOTAL,'page_count':TOTAL,'pdf_sha256':sha(staged),
                   'stage_details':{'pages':[50,51,52,53,54],
                       'parent_pdf_sha256':sha(backup),'original_49_page_bodies_preserved':True,
                       'original_images_preserved':True,'original_test_CSV_unchanged':True,
                       'legacy_English_punctuation_glyphs_repaired':repaired,
                       'generator_sha256':sha(__file__),'layout':layout,
                       'text_bounds_verified_pages':bounds,'training_or_inference':False,
                       'visual_review':'pending'},
                   'visual_review':'original 49-page visual QA retained; new appendix and navigation panels pending visual inspection',
                   'updated_utc':datetime.now(timezone.utc).isoformat()})
        shutil.copyfile(staged,pdf)
        (OUT/(base+'.qa.json')).write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
        (OUT/(base+'.sources.stages.json')).write_text(json.dumps(source_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        (OUT/(base+'.source.txt')).write_text('\n\f\n'.join(page.extract_text() for page in updated.pages),encoding='utf-8')
        print(json.dumps({'language':lang,'pages':TOTAL,'appendix_layout':layout,
                          'original_body_and_images_preserved':True,'sha256':sha(pdf)},ensure_ascii=False))


if __name__=='__main__': main()
