"""Close a failed admission audit without pretending Gaussian training occurred."""
from pathlib import Path
from collections import Counter
import numpy as np
from variants.staged_reconstruction.i10_pseudo_depth import io


def main():
    io.verify();a=io.read(io.REPORT/'audit.json');s=a['summary'];p=io.read(io.BASE/'protocol.json')
    assert not s['local_evidence_gate_passed'],'Passing admission requires separate training qualification'
    for name in ['qualification.json','independent_replay.json','independent_audit.json']:
        assert io.read(io.REPORT/name)['passed'],name
    figures=io.read(io.REPORT/'figures.json');assert len(figures['images_sha256'])==6
    for path,h in figures['images_sha256'].items():assert io.sha(io.ROOT/path)==h,path
    manual=dict(actual_visual_review=True,actual_reviewed_images=6,
        reviewed_images_sha256=figures['images_sha256'],
        findings=[
            '实际查看固定训练index0/24/72/119的RGB、原始相对逆深度、仿射深度展示和稀疏目标，以及完整三时段覆盖图。',
            '固定0/24/72/119均未过标定：分别分半拟合退化、锚点几何退化、锚点不足、锚点或尺度不一致；不是固定86/188新训练图。',
            '附加index28是预定四图以外按最小通过索引确定的展示样本，仅作事后可视化；通过局部标定，仍无可用连续动态深度目标，没有用于修改门槛。',
            '锚点多位于立柱和底座边缘，未覆盖全部鸭子可见表面；这里由RGB三角化相容性产生，不按蓝色选点，不赋静态身份。',
            'Small在原黑背景中也给出连续有限相对深度，形状排序可辨但不能直接解释为可靠表面。未将背景颜色用于筛选，未声称几何GT误差。',
            '所有图右下均无绿色训练目标；左下黑色是无标定/无有效展示值，不是零误差；即使显示完整仿射深度，也不代表整图可监督。'
        ],independent_geometry_GT=False,new_training_results=False,historical_color_parent_disclosed=True)
    io.write(io.REPORT/'manual_review.json',manual)
    motions=[q for q in a['motion'] if 'row' in q]
    local=Counter(reason for q in motions for reason in q['local_reasons'])
    maximum=max(q['valid'] for q in motions)
    supplemental=dict(motion_reasons=dict(Counter(q['reason'] for q in a['motion'])),
        local_observation_reasons=dict(local),all_native_motion_observations=len(motions)*12,
        maximum_valid_observations_one_track=maximum,
        per_track_valid_histogram=dict(Counter(str(q['valid']) for q in motions)),
        motion_numeric_check_reached=sum('motion_P90_relative' in q for q in motions),
        boundary_rates={key:{'denominator':v['observations'],
            **{name:v[name]/v['observations'] if v['observations'] else None
               for name in ['learned_visibility_failure','bidirectional_disagreement','track_rejected']}}
                       for key,v in a['boundary_groups'].items()},
        all_observations_preserved=True,optimizer_updates=0)
    coefficient=[q for q in a['calibration'] if q['coefficients'] is not None]
    for name,column in [('affine_slope',0),('affine_intercept',1)]:
        x=[q['coefficients'][column] for q in coefficient]
        supplemental[name+'_quantiles']={str(t):float(np.quantile(x,t)) for t in [0,.05,.5,.95,1]}
    for name in ['check_median_relative','check_P90_relative','split_median_relative','split_P90_relative']:
        x=[q[name] for q in a['calibration'] if name in q]
        supplemental[name]=dict(denominator=len(x),median=float(np.median(x)),P90=float(np.quantile(x,.9)))
    io.write(io.REPORT/'diagnostic_accounting.json',supplemental)
    lines=['# L7-P：Rubber Duck伪深度准入结果','',
        '结论：当前固定Small先验未形成满足协议的连续动态深度目标，不进入Gaussian续训，保留J6。这里主要失败于尺度标定与时间/空间支撑不足；没有足够合格观测进入运动一致性的数值检验，不能把零准入解释为追踪或运动补偿已被证明无效。','',
        'bf8fda85在原始伪深度输出前提交协议、17项测试及456文件级别冻结（确切数量见source_freeze.json）。原生PIDG和历史权重不改；无新颜色阈值、语义/静态掩膜、物体数量、GT深度法线或物理更新。J6历史颜色条件来源仍披露。','',
        '## 实际执行','',
        '- 全120张训练RGB，400×225观察分辨率，官方固定DA-V2 Small、518预处理、FP32 eval，无模型替换或微调。原PIDG的RGBA到黑背景RGB合成保持，alpha没有用于选锚点或分类。',
        '- 原362条RGB静止相容轨迹三角化只作尺度假设；每图4px格去重、8px空间fit/check隔离、正斜率稳健仿射逆深度。双子集拟合分别检查尺度稳定性；训练候选仅允许fit锚点凸包内插值。',
        '- 原222条运动候选全部保留，其中122条通过原T4绑定/互反/自身可见性资格。将伪深度反投影到世界坐标，再减去固定J6的时变位移；偶数6观测拟合、奇数6观测检查，至少各4有效。父运动不是GT，原T4全局覆盖失败保持。',
        '- 原4433条提议×12时刻=53196观测全部纳入5×5深度突变边界代理统计，包含不可见、反向不一致和原被拒提议；未知不算背景/动态/零误差。','',
        '| 指标 | 实测 | 预声明局部准入 |','|---|---:|---:|',
        f'| 标定通过视图 | {s["calibrated_views"]}/120 | ≥30 |',
        f'| 早/中/晚标定视图 | {s["calibrated_views_by_third"]} | 各≥8 |',
        f'| 连续合格运动轨迹 | {s["reliable_motion_tracks"]} | ≥30 |',
        f'| 早/中/晚合格轨迹 | {s["reliable_tracks_by_third"]} | 各≥5 |',
        f'| 不同动态高斯行 | {s["distinct_rows"]} | ≥25 |',
        f'| 最终偶数帧深度目标 | {s["auxiliary_observations"]} | 仅可靠观测 |','',
        '局部门槛只针对新弱深度试验，不是全物体覆盖/几何晋级，也未覆盖或改判旧T4/L4/L5/L6失败。','',
        '## 尺度与时间支撑','',
        '| 未通过原因 | 视图数/120 |','|---|---:|']
    for name,count in s['calibration_reasons'].items():lines.append(f'| {name} | {count} |')
    lines+=['',f'122条原生合格绑定共1464个观测：未标定{local["uncalibrated"]}、凸包外{local["outside_fit_hull"]}、局部可用{local["passed_local_interpolation"]}。单轨最多{maximum}/12个有效深度，所有122条均不足fit/check各4，因此运动补偿后的数值一致性检验实际到达0条。没有伪造相机补偿前后误差下降；不能据此断言动态深度一定不准确。',
        '仿射斜率/截距分布、可检查视图的held和分半稳定性分母见diagnostic_accounting.json。原始系数随图变化，不能单独称为物理尺度漂移；本轮依据空间held及双拟合的实际深度差判断。多数尺度锚点集中于立柱/底座，局部标定通过也不允许将尺度可靠性外推到远离锚点的鸭子表面。','',
        '## 遮挡边界代理审计','',
        '| 分组 | 观测分母 | 学习追踪可见性失败 | 前后不一致 |','|---|---:|---:|---:|']
    for key,q in a['boundary_groups'].items():
        lines.append(f'| {key} | {q["observations"]} | {q["learned_visibility_failure"]} ({100*q["learned_visibility_failure"]/q["observations"]:.2f}%) | {q["bidirectional_disagreement"]} ({100*q["bidirectional_disagreement"]/q["observations"]:.2f}%) |')
    lines+=['','42276/53196个观测缺少可标定边界诊断，不是不存在遮挡。在有限可检查子集中，突变边界组可见性失败率并未高于内部组；没有证据把边界/遮挡当成唯一或主要失败原因。深度突变与学习可见性均为代理，不是遮挡GT。','',
        '## 弱约束与训练状态','',
        '默认关闭的稳健log深度损失已实现：共享权重0.001，误差尺度0.05，目标/置信度停止梯度；未知暂停、两次矛盾撤销、两次通过恢复。17项单测包含尺度退化、held不回灌、空间去重、相机旋转平移往返、运动补偿、未知/边界、有限梯度和flags-off对象一致。',
        '但当前目标为0，故本轮没有集成进原生训练，没有新增optimizer更新、PSNR改进、86/188图、PLY或训练视频，也没有声称通过新候选原生梯度/双Adam恢复资格。只有准入通过后另立匹配协议才可开始训练。','',
        '## 实际看图与独立核验','']+['- '+x for x in manual['findings']]
    lines+=['','独立进程重新加载Small，固定0/24/72/119原始数组精确重放；全120输入/CPU决策/原始数组SHA重放一致。另一套独立计算核验所有120图锚点去重、64组仿射系数、122组世界反投影及全部53196边界观测/分母。父模型和冻结数据源码SHA不变，0次优化。',
        '保留失败记录：首次冻结读取旧版扁平receipt报KeyError，在推断前修复；独立评估1误将有限精度相机旋转转置当精确逆（正交误差约3.63e-7），独立v2改线性求解，保留1e-10校验容差；绘图1全90000像素remap超OpenCV上限，独立绘图v2在整数格直接计算仿射逆深度。冻结推断/判定代码、权重及门槛均未改。','',
        '下一步应先检验跨帧联合尺度标定和更广空间锚点覆盖，保留遮挡/未知；当前结果不支持直接提高深度权重、整图自监督或增加训练时长。单个Small准入失败也不能证明RGB数据不可解。','',
        '最终状态：completed_audit_no_admissible_depth；保留J6，不启动训练、不自动换模型/扫参/放宽门槛。']
    # Exact file count replaces descriptive approximation above.
    count=len(io.read(io.REPORT/'source_freeze.json')['files'])
    lines=[line.replace('456文件级别冻结',f'{count}文件冻结') for line in lines]
    (io.REPORT/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    result=dict(status='completed_audit_no_admissible_depth',pending=False,pending_review=False,
        optimizer_updates=0,current_model_replaced=False,admission_passed=False,
        weak_loss_default_off_implemented=True,native_training_integration_qualified=False,
        motion_numeric_consistency_test_reached=0,calibrated_views=24,reliable_motion_tracks=0,
        independent_3D_GT=False,historical_color_parent_disclosed=True,no_automatic_sweep=True)
    names=['assessment.md','manual_review.json','diagnostic_accounting.json','audit.json','figures.json',
           'source_freeze.json','qualification.json','independent_replay.json','independent_audit.json',
           'independent_evaluation_freeze.json','independent_evaluation_freeze_v2.json','figure_evaluation_freeze_v2.json',
           'independent_failure_attempt1.json','figure_failure_attempt1.json']
    io.write(io.REPORT/'final_review.json',dict(**result,artifacts_sha256={name:io.sha(io.REPORT/name) for name in names}))
    for path in [io.REPORT/'final_status.json',io.RUNS/'final_status.json',io.RUNS/'state.json',io.RUNS/'active.json',io.RUNS/'pipeline.json']:io.write(path,result)
    with (io.BASE/'progress.md').open('a',encoding='utf-8',newline='\n') as f:
        f.write('\n- All120 raw predictions, independent reload/CPU and separate arithmetic replay complete;17 tests,6 actual reviewed images.24 calibrated views,0 admissible motion tracks,0 Gaussian updates. See v1/assessment.md; J6 retained.\n')
    print('L7_FINAL_REVIEW',result)


if __name__=='__main__':main()
