"""Existing experiment images for the dated weekly PDF, without model inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math
from PIL import Image, ImageDraw, ImageFont, ImageChops

ATLAS_PAGE_COUNT = 15

def prepare_atlas(root, snap):
    root, snap = Path(root), Path(snap)
    out=snap/'atlas'; out.mkdir(parents=True,exist_ok=True)
    records={}; transforms=[]
    font=ImageFont.truetype('C:/Windows/Fonts/Dengb.ttf',33)
    def track(rel):
        source=root/rel
        target=snap/'image_sources'/rel
        if not target.exists():
            data=source.read_bytes()
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
        data=target.read_bytes()
        records[rel]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
                      'snapshot':str(target.relative_to(snap))}
        return target
    def image(rel,side=None):
        with Image.open(track(rel)) as im: im=im.convert('RGB')
        if side is not None:
            # Standard exports have a 50px title; native diagnostic pairs do not.
            assert im.size in ((3200,950),(3200,900)),(rel,im.size)
            title_height=im.height-900
            im=im.crop((1600*side,title_height,1600*(side+1),im.height))
        return im
    def copy(rel,name):
        im=image(rel); im.save(out/(name+'.png'))
        transforms.append({'output':'atlas/'+name+'.png','source':rel,'operation':'unmodified existing figure'})
        return 'atlas/'+name+'.png'
    def grid(name,panels,cols=4,ref=None,crop=None):
        # panel = (label, existing path, pair side or None).
        ims=[image(rel,side) for _,rel,side in panels]
        assert len({im.size for im in ims})==1,[(panels[i][0],im.size) for i,im in enumerate(ims)]
        if crop is None and ref is not None:
            ref_im=image(ref[0],ref[1])
            channel=ImageChops.lighter(ImageChops.lighter(*ref_im.split()[:2]),ref_im.split()[2])
            bounds=channel.point(lambda v:255 if v>18 else 0).getbbox()
            assert bounds
            x0,y0,x1,y1=bounds
            pad=45
            crop=(max(0,x0-pad),max(0,y0-pad),min(ref_im.width,x1+pad),min(ref_im.height,y1+pad))
        if crop is None: crop=(0,0,*ims[0].size)
        ims=[im.crop(crop) for im in ims]
        cellw=520; cellh=round(cellw*ims[0].height/ims[0].width); title=46; gap=12
        sheet=Image.new('RGB',(cols*cellw+(cols-1)*gap,math.ceil(len(ims)/cols)*(cellh+title+gap)-gap),'#eef3f5')
        draw=ImageDraw.Draw(sheet)
        for i,(im,(label,rel,side)) in enumerate(zip(ims,panels)):
            x=(i%cols)*(cellw+gap); y=(i//cols)*(cellh+title+gap)
            draw.rectangle((x,y,x+cellw,y+title),fill='#183b50')
            draw.text((x+13,y+8),label,fill='white',font=font)
            sheet.paste(im.resize((cellw,cellh),Image.Resampling.LANCZOS),(x,y+title))
        sheet.save(out/(name+'.png'))
        transforms.append({'output':'atlas/'+name+'.png','panels':[{'label':a,'source':b,'pair_side':d} for a,b,d in panels],
                           'common_crop_xyxy':list(crop),'resize':'same crop for every model; proportional Lanczos; no image repair'})
        return 'atlas/'+name+'.png'

    assets={}
    assets['coverage_rgb']=copy('output/pdf/coverage_assets/frame153_rgb.png','coverage_rgb')
    assets['coverage_alpha']=copy('output/pdf/coverage_assets/frame153_alpha.png','coverage_alpha')
    assets['color_subject']=copy('output/pdf/color_repair_assets/frame_00038_subject.png','color_subject')
    assets['color_roi']=copy('output/pdf/color_repair_assets/frame153_color_roi.png','color_roi')
    assets['color_blue']=copy('output/pdf/color_repair_assets/frame153_blue_defect.png','color_blue')
    assets['centroid']=copy('experiments/centroid_tuning/analysis/frame0049.png','centroid49')
    iso='runs/pidg_v3_iso_scratch/render_comparisons'
    for frame in (1,85):
        fn=f'{(frame-1)//4:05d}.png'
        control=f'{iso}/V3_scratch_control/pairs/50000/{fn}'
        assets[f'iso{frame}']=grid(f'iso{frame}',[
            ('GT',control,0),('V3 control',control,1),('V3 + iso .001',f'{iso}/V3_scratch_iso_1e3/pairs/50000/{fn}',1)],3,(control,0))
    integrity='runs/pidg_integrity_continuation/render_exports/final50000'
    names=[('C1', 'C1_reused'),('H1 coverage','H1_balanced_coverage'),('H2 strain','H2_temporal_cohesion'),
           ('H3 combined','H3_combined'),('H4 density','H4_color_density'),('F1 feedback','F1_feedback'),('F2 retry','F2_conservative_retry')]
    for frame in (85,153):
        fn=f'{(frame-1)//4:05d}.png'; gt=f'{integrity}/gt/{fn}'
        panels=[('GT',gt,None)]+[(lab,f'{integrity}/{directory}/renders/{fn}',None) for lab,directory in names]
        assets[f'integrity{frame}']=grid(f'integrity{frame}',panels,4,(gt,None))
    deform='runs/pidg_deformation_repair/render_exports/final50000'
    gt=f'{deform}/gt/00021.png'
    assets['deformation']=grid('deformation85',[('GT',gt,None)]+
        [(lab,f'{deform}/{directory}/renders/00021.png',None) for lab,directory in
         [('H3','H3_reused'),('D1 affine','D1_motion_affine'),('D2 covariance','D2_covariance_continuity'),('D3 combined','D3_combined')]],5,(gt,None))
    material='runs/pidg_c1_material/render_exports/final50000'; gt=f'{material}/gt/00038.png'
    assets['material']=grid('material153',[('GT',gt,None)]+[(lab,f'{material}/{directory}/renders/00038.png',None) for lab,directory in
       [('P0 control','P0_C1_control'),('P1 fixed E','P1_fixed_E'),('P2 learned E','P2_learn_E')]],4,(gt,None))
    obj='runs/pidg_object_id/render_exports/final50000'; gt=f'{obj}/gt/00021.png'
    assets['objectid']=grid('objectid85',[('GT',gt,None)]+[(lab,f'{obj}/{directory}/renders/00021.png',None) for lab,directory in
       [('C1 control','O0_C1_control'),('Frozen ID','O1_frozen_ID'),('Learned ID','O2_selfsup_ID')]],4,(gt,None))
    assets['velocity']=copy('experiments/frozen_velocity_probe/velocity_fit_diagnostics.png','velocity')
    pair=lambda folder:f'{folder}/pairs/best/00021.png'
    k1=pair('output/flow_comparison/K1_flow_independent')
    assets['J_K_L']=grid('J_K_L85', [('GT',k1,0),
       ('J1 monotonic',pair('output/monotonic_training/J1_monotonic_kinematic'),1),
       ('K1 independent',k1,1),('K2 trajectory',pair('output/flow_comparison/K2_flow_trajectory'),1),
       ('L0 corrected',pair('output/unified_flow/L0_corrected_trajectory'),1),('L1 shared stress',pair('output/unified_flow/L1_shared_stress'),1)],3,(k1,0))
    c1=pair('output/c1_flow_ablation/C1_repaired')
    assets['M_N_flow']=grid('M_N_flow85',[('GT',c1,0),('C1 control',c1,1),
       ('C1 + flow',pair('output/c1_flow_ablation/O1_C1_flow_on'),1),
       ('M1 gradient guard',pair('output/momentum_repair/M1_guarded_shared'),1),
       ('N1 coherent',pair('output/coherent_momentum/N1_coherent_momentum'),1)],3,(c1,0))
    assets['momentum']=copy('output/coherent_momentum/rollout_displacement_3d.png','momentum')
    assets['shape']=copy('output/shape_coherence/gate_review.png','shape')
    assets['support']=copy('output/surface_support/center_review.png','support')
    assets['shortclip']=copy('output/short_clip_diagnostic/native_review.png','shortclip')
    assets['motion_forecast']=copy('output/motion_correspondence/forecast_examples.png','motion_forecast')
    assets['surface_timeline']=copy('output/surface_correspondence/support_timeline.png','surface_timeline')
    assets['point_objects']=copy('output/object_membership/point_objects.png','point_objects')
    assets['association_rgb']=copy('output/association_training/rgb_comparison.png','association_rgb')
    assets['association_groups']=copy('output/association_training/groups_comparison.png','association_groups')
    assets['transfer_early']=copy('output/transfer_integrity/early_native_inspection.png','transfer_early')
    assets['transfer_crop']=copy('output/transfer_integrity/native_crop_fixed86_t2.png','transfer_crop')
    assets['scratch_crop']=copy('output/scratch_integrity/native_crop_fixed86_t2.png','scratch_crop')
    assets['static_pool']=copy('output/static_pool/review/fixed_0086_native_crops.png','static_pool')
    assets['S1']=copy('output/staged_reconstruction/S1_static/S1_review.png','S1')
    # Observed t86 GT is taken from an existing native GT|Render export.
    gt86='output/association_training/T1_association/native_GT_Render0086.png'
    stages=[('S2 first','S2_objects'),('S2 repair 1','S2_recovery'),('S2 repair 2','S2_geometry_recovery')]
    assets['S2_observed']=grid('S2_observed86',[('GT t86',gt86,0)]+[(label,f'output/staged_reconstruction/{d}/original_t0086_native.png',None) for label,d in stages],4,(gt86,0))
    assets['S2_fixed']=grid('S2_fixed86_t2',[(label,f'output/staged_reconstruction/{d}/fixed86_t0002_native.png',None) for label,d in stages],3)
    assets['S2_fixed_late']=grid('S2_fixed86_t188',[(label,f'output/staged_reconstruction/{d}/fixed86_t0188_native.png',None) for label,d in stages],3)
    stages=[('Stage S3','S3'),('Stage S4','S4'),('Stage S5','S5')]
    assets['S345_observed']=grid('S345_observed86',[('GT t86',gt86,0)]+[(label,f'output/staged_reconstruction/{d}/original_t0086_native.png',None) for label,d in stages],4,(gt86,0))
    for frame in (2,188):
        assets[f'S345_fixed{frame}']=grid(f'S345_fixed86_t{frame}',[(label,f'output/staged_reconstruction/{d}/fixed86_t{frame:04d}_native.png',None) for label,d in stages],3)
    manifest={'prepared_utc':datetime.now(timezone.utc).isoformat(),'data_cutoff':'same 2026-11-03 11:07 UTC experiment snapshot; existing exports only',
       'sources':records,'derived_figures':transforms,'figure_paths':assets,
       'selection':'Previously discussed frames 1/49/85/86/153/188; diagnostic examples, not representative accuracy estimates.',
       'no_model_inference_or_training':True}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return assets,manifest

def render_atlas(assets,start,p,fig,box):
    n=2
    def image(key,h,caption):
        nonlocal n
        n+=1; fig(assets[key],h,f'图{n}  '+caption)
    def page(code,title,sub): start(f'图像结果 {code} / {ATLAS_PAGE_COUNT}',title,sub)

    page('01','覆盖、邻域与增密：RGB与alpha','对应正文p3；C0-C5 final50k，同一测试原帧153。沿用旧报告的GT统一裁剪。[02]')
    image('coverage_rgb',315,'GT与C0-C5的RGB。上排GT/C0/C1/C2，下排C3/C4/C5/C1误差；误差色标[0,0.25]。旧C1的局部偏绿明显，说明高轮廓IoU仍可能伴随颜色退化。')
    image('coverage_alpha',220,'同一帧的完整alpha，保留图内模型标签和统一[0,1]显示范围。背景RGB黑不代表alpha为零；也不能将高alpha暗斑计成透明缺口。')
    p('这两张图展示旧版覆盖消融；下一页才是颜色梯度修复后的C1-C3，命名和阶段不可混用。','small')

    page('02','颜色修正：主体与局部细节','对应正文p3；测试原帧153，final50k。[03]')
    image('color_subject',250,'从左到右GT / 修复后C1 / C2 / C3；所有面板采用GT决定的相同主体裁剪。三组基本恢复黄色，接触区和障碍物仍有局部缺陷。')
    image('color_roi',145,'相同ROI，左至右GT / 旧C1 / 新C1 / 新C2 / 新C3。旧C1红通道缺失，新版不再呈同样偏绿；这是组合修正的局部证据。')
    image('color_blue',140,'蓝色区域相同裁剪：GT / 新C1 / 新C2 / 新C3。残留高alpha暗斑显示，颜色修好不等于形状和遮挡均正确。该ROI为历史事后诊断，非随机抽样。')

    page('03','各向同性与质心：改善并不均匀','对应正文p3；各向同性同step0训练50k，质心为其独立G0/G1对照。[01,04]')
    image('iso1',140,'原帧1：GT / 无iso / iso=0.001；来自final50k。只按GT边界统一裁黑背景，未分别调整各模型的显示范围。')
    image('iso85',140,'原帧85，相同三列。整图布局接近时，局部轮廓差异与均值指标仍需共同判断；不能仅凭选帧宣布全面改善。')
    image('centroid',245,'原帧49，沿用原分析图中的GT/G0/G1标签与共同裁剪。此帧为G1明显退化的例子；全局质心对齐仍不能保证局部鸭体形态正确。')

    page('04','H/F完整度消融：两张困难帧','对应正文p4；同一50k终态；每组图中GT和七模型顺序一致。[05,06]')
    image('integrity85',270,'原帧85。上排GT / C1 / H1 / H2；下排H3 / H4 / F1 / F2。该帧H3相对C1明显退化，保留它以显示平均收益的边界。')
    image('integrity153',270,'原帧153，模型顺序同上。每个原帧均由该帧GT独立决定共同裁剪，同帧所有模型完全一致；不同帧的取景比例不用于大小/运动量比较。')
    p('H3的均值改进与关键坏帧可以同时成立；这些图是已知问题帧的解释性展示，不取代50帧统计。','small')

    page('05','形变、材料与物体分组消融','对应正文p4-5。三个实验族各有自己的匹配对照，跨行不能当作单因素比较。[07,09,10]')
    image('deformation',160,'原帧85：GT / H3 / D1 / D2 / D3，final50k。D1局部恢复不等于最差帧与全局指标全面改善。')
    image('material',190,'原帧153：GT / P0匹配对照 / P1固定E / P2可学E。固定与可学材质均未优于本实验对照；图中形状/颜色变化不能解释成真实刚度估计。')
    image('objectid',170,'原帧85：GT / C1控制 / 冻结ID / 可学ID，final50k。可学分组没有稳定消除局部轮廓与接触伪影，槽编号不等于真实物体或材料身份。')

    page('06','速度诊断与J/K/L图像结果','对应正文p5-6；冻结速度诊断与新的重建训练分开展示。[11-14]')
    image('velocity',250,'冻结H3轨迹的速度拟合诊断原图。目标是模型轨迹导数，非物理速度GT；单调时钟改善时间拟合，但跨ID泛化仍差。图内保留各臂及坐标定义。')
    image('J_K_L',290,'测试原帧85、各自验证选出的best。上排GT / J1 / K1；下排K2 / L0 / L1。使用相同GT裁剪，缺失鸭体与拉丝可见。K到L还修了辅助流程，跨模型差异不能全归因于共享应力。')

    page('07','动量修正与C1光流消融','对应正文p7。原帧85按同一GT裁剪；模型快照均为各自验证选择best。[15-17]')
    image('M_N_flow',290,'上排GT / C1 / C1加光流；下排M1 / N1。空余格仅为排版，不是未渲染的模型。各实验族的对照关系见正文；不将本拼图当统一架构排名。')
    image('momentum',235,'M1/N1共同点的物理推进与渲染位移原诊断图，保留图内帧、坐标和图例。它比较同一内部残差的相关量，不是独立真实3D运动验证。')
    p('图像恢复、观测运动改善和物理残差下降是不同证据。N1仍未通过跨时刻动量25%改善门槛。','small')

    page('08','形态准入与局部支持诊断','对应正文p8；P1只训练到16k，形态loss尚未启用。[18,19,22]')
    image('shape',350,'P1 16k准入图：从上至下为原训练帧2/74/86/108，左GT、中候选16k、右固定参考点库（未知变暗）。中晚期RGB由未知点承担，早期标记贡献消失。形态loss尚未启用，不能作为形态损失训练后的消融。')
    image('support',145,'冻结C1/P1的中心支持原图：对照投影中心与实际可见足迹。点仍在画面内，不代表它仍支撑当前鸭子表面。图内保留模型与时刻标签。')
    page('09','短片段与运动历史的可视化','对应正文p8；训练图像诊断与人工隐藏观测的预测试验，无Gaussian重训。[20,22]')
    image('shortclip',170,'短片段原生复核图。相同C1在训练分辨率的清晰片段可重建较好；低分辨率直接渲染的差异不能当作模型新退化。该图为原训练观测，不是新增测试。')
    image('motion_forecast',350,'运动历史预测示例：原返回帧14/83/154，保留原图例；叉号为重现前预测，圆圈为伪参照。同一行颜色仅对应该窗口的局部种子。位置预测接近不代表自然完全遮挡后的身份已确认。')
    p('这些窗口有筛选偏差且观测相关；它们是短期二维预测的诊断图，不是真实三维材料点运动。','small')

    page('10','可见表面点库与局部物体归属','对应正文p8；均为冻结模型诊断，新增Gaussian优化步数为0。[21,23]')
    image('surface_timeline',265,'16个固定点库的支持时间线及接受贡献覆盖。选点、通过、可见但失败、身份未知分别记录；平均17.76%是严格门槛下接受覆盖率，不是模型正确率。')
    image('point_objects',275,'局部物体与点中心叠图，保留原帧和身份域标注。不同片段建立局部身份；同色跨域不自动等同。历史归属和当前可见支持分开保存。')
    p('未覆盖区域保持未知；没有根据最近鸭子、空间距离或行号重叠补真实身份。','small')

    page('11','关联续训：RGB与固定点库贡献','对应正文p9；原C1 / T0普通续训 / T1关联续训，图内列顺序和原帧标签保持原样。[24]')
    image('association_rgb',270,'早/中/晚四个选定训练时刻的RGB对照。训练图像外观接近，不能据此断言遮挡后身份恢复。')
    image('association_groups',270,'相同时刻的固定早期点库贡献对照。颜色只编码已注册点库；晚期旧库没有恢复鸭子区域支撑。局部纯度改善与总体支持缩小同时存在。')

    page('12','两版几何约束：原生局部图','对应正文p11-12；首版U0/U1续训、第二版S0/S1从零。固定相机86的新组合没有GT。[27,28]')
    image('transfer_early',190,'首版固定相机86、时刻2：左历史C1、中U0、右U1；上排RGB、下排alpha。三组均为无独立GT的新相机/时间组合，仍有明显拉丝与碎片。')
    image('transfer_crop',190,'首版固定86、时刻2的同坐标原生裁剪，左U0、右U1。局部仍有拉丝和分层；来自原渲染像素，未补洞。')
    image('scratch_crop',170,'第二版同固定86、时刻2，左S0、右S1。各版内裁剪条件匹配；两版种子/绑定不同，不作像素差因果比较。更早接入局部约束仍未解决该伪影。')

    page('13','静态池负结果与独立静态重建','对应正文p13；R0与Stage S1的架构和相机不同，不是匹配消融。[29-31]')
    image('static_pool',365,'R0固定86的原生同坐标裁剪，图内标明H0/H1和时刻。两组早晚仍碎裂；数值审计另外显示硬静态子集被透明化，不能将零时变误差当作成功几何。')
    image('S1',190,'独立Stage S1在t2/86/188的RGB/静态/alpha复核原图，4126点、12k更新。只重建静态，鸭子未渲染不计作缺失；边缘与几何仍有不确定性。')

    page('14','Stage S2：两次修正的图像证据','对应正文p14；每次16k，点数/网络预算保留。静态S1几何固定。[32]')
    image('S2_observed',200,'实际相机、训练原帧86：GT / S2首版 / 第一次修正 / 第二次修正。由GT统一裁主体；修正改善观测动态覆盖，但不是完整新视角重建。')
    image('S2_fixed',185,'固定相机86、时刻2：首版 / 第一次修正 / 第二次修正。保留完整1600×900视野后等比缩放，未用GT裁掉异常支撑；此组合没有独立GT。')
    image('S2_fixed_late',185,'固定相机86、时刻188，同一模型顺序和完整视野。数值门槛通过仍伴随扁片、拉丝和碎片，故只允许条件接续。')

    page('15','Stage S3-S5：观测好于固定新组合','对应正文p14；S3 +6k、S4 +4k、S5 +4k；快照仍截止11/3 11:07。[33-36]')
    image('S345_observed',200,'实际相机、原帧86：GT / S3 / S4 / S5，按GT共同裁主体。观测轮廓可识别，仍有局部模糊。')
    image('S345_fixed2',185,'固定相机86、时刻2：S3 / S4 / S5，完整视野。鸭体局部颗粒、条纹和边缘失真仍在，静态结构相对稳定；没有该固定组合的独立GT，不计算PSNR。')
    image('S345_fixed188',185,'固定相机86、时刻188，同一顺序和完整视野。S5条件先验全程未激活，不能把图像变化解释为碰撞修复；S6正式终态与匹配图像见后续补充页。')

    return n-2
