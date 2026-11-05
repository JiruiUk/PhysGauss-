"""Close this one human-authorized trial after actual image review and audits."""
import argparse
from datetime import datetime,timezone
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--finalize-i10-joint',action='store_true')
    if not parser.parse_args().finalize_i10_joint:raise SystemExit('J6 finalization default off')
    b=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_joint';report=b/'v1'
    runs=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_joint/v1'
    out=ROOT/'output/staged_reconstruction/RubberDuck_I10_joint/v1'
    ev=read(report/'evaluation.json');videos=read(report/'videos.json')
    assert read(report/'independent_audit.json')['passed']
    names=[]
    for cam in [86,188]:
        for time in [2,100,200]:
            names += [f'c{cam:03d}_t{time:03d}_joint_comparison.png',f'c{cam:03d}_t{time:03d}_decomposition.png',
                      f'c{cam:03d}_t{time:03d}_joint_depth_trust_jointRGB_native.png']
    names += [f'observed_train_index{i:03d}.png' for i in [0,24,48,72,96,119]]
    names += ['c086_t200_joint_depth_trust_dynamicAlpha_native.png','c188_t100_joint_depth_trust_dynamicAlpha_native.png',
              'c086_t200_joint_depth_trust_dynamicD-alpha_native.png','c188_t100_joint_depth_trust_staticD-alpha_native.png']
    names += [f'fixed_c{c:03d}_parent_control_joint.decoded.png' for c in [86,188]]
    assert len(names)==30
    manual={'actually_viewed':True,'images':{(out/name).relative_to(ROOT).as_posix():sha(out/name) for name in names},
        'static_visual_nonregression_in_sampled_views':True,
        'clear_alpha_depth_cleanup_and_dynamic_completeness':False,
        'observations':[
            'Static blue main bodies remain present; sampled images show no obvious new large disappearance versus accepted J4 static source, while edge haze and uneven faces remain.',
            'Observed train trajectories generally fit more cleanly after full-RGB continuation, including static boundary and dynamic appearance.',
            'Fixed c86/t2 remains fragmented; c86/t200 retains stretched yellow fragments, a long stream and dark RGB support with broad bright dynamic alpha and finite D/alpha.',
            'Fixed c188/t100 retains stretched and detached dynamic parts with a horizontal high-alpha plume; c188/t200 is more duck-like in RGB but diagonal alpha support persists.',
            'Joint appearance changes are modest in these off-trajectory comparisons. Improved full-image PSNR does not establish geometric cleanup.'
        ],'decoded_contacts_actually_viewed':True,'actual_browser_playback_checked':False,
        'sampled_review_not_independent_GT_or_exhaustive_proof':True}
    write(report/'manual_review.json',manual)
    means=ev['validation']['means'];delta=means['joint_depth_trust']['PSNR']-means['frozen_static']['PSNR']
    status='completed_not_promoted';stamp=datetime.now(timezone.utc).isoformat()
    final={'status':status,'pending':False,'pending_review':False,'finished_at_UTC':stamp,
        'formal_updates_per_arm':2000,'formal_updates_total':4000,
        'user_requested_I10_joint_training_executed':True,'old_paused_jobs_resumed':False,
        'PSNR_nonregression_gate':ev['validation']['PSNR_nonregression_gate'],
        'joint_minus_matched_control_PSNR_dB':delta,
        'static_visual_nonregression_sampled':True,'dynamic_alpha_depth_cleanup_completeness_gate':False,
        'independent_artifact_audit':True,'independent_geometry_GT':False,
        'automatic_extension':False,'old_failure_and_nonpromotion_records_preserved':True,
        'evidence':{name:sha(report/name) for name in ['evaluation.json','videos.json','manual_review.json','independent_audit.json','recovery.json']}}
    write(report/'final_review.json',final);write(report/'final_status.json',final)
    write(b/'progress.json',final)
    write(runs/'final_status.json',final)
    pipeline=read(runs/'pipeline.json');pipeline.update(status=status,pending_review=False,
        training_end_status='trained_pending_evaluation',finished_at_UTC=stamp)
    write(runs/'pipeline.json',pipeline)
    write(runs/'active.json',{'status':status,'pending_review':False,'running':False,'formal_updates_total':4000})
    for phase in ['qualification','formal']:
        for arm in ['frozen_static','joint_depth_trust']:
            path=runs/phase/arm/'final_status.json';row=read(path)
            row.update(training_end_status=row['status'],status='qualified' if phase=='qualification' else status,pending_review=False)
            write(path,row)
    text=f'''# J6：I10 动态模型与已接受静态背景的联合训练

已完成用户要求的 Rubber Duck 联合训练：静态冻结对照和静态透明度/SH 联合组各新增 2000 步，共 4000 次正式更新。联合组的完整图像 RGB 指标改善，但固定视角仍有动态碎裂、拉丝和暗色高 alpha 拖尾，清理/完整度人工门槛未通过；状态为 `completed_not_promoted`。本轮执行已完成，原模型与全部结果保留，不自动延长或恢复旧任务。

| 原生 1600 宽，全 30 张 val | 平均 PSNR (dB) | 平均 SSIM | 平均 L1 |
|---|---:|---:|---:|
| I10 动态 + J4 静态组合起点 | {means['I10_J4_initial']['PSNR']:.6f} | {means['I10_J4_initial']['SSIM']:.6f} | {means['I10_J4_initial']['L1']:.8f} |
| 静态冻结，动态续训 2000 步 | {means['frozen_static']['PSNR']:.6f} | {means['frozen_static']['SSIM']:.6f} | {means['frozen_static']['L1']:.8f} |
| 静态外观与动态联合 2000 步 | {means['joint_depth_trust']['PSNR']:.6f} | {means['joint_depth_trust']['SSIM']:.6f} | {means['joint_depth_trust']['L1']:.8f} |

联合较匹配对照 +{delta:.6f} dB，通过原最大下降 0.3 dB 的 RGB 门槛。这是全图 RGB，未分离出独立静态/动态 GT；不能据此声称物体几何完整。起点标签明确指组合模型，不是独立 I10 或原 S3-U。历史 I10 原生 CUDA 轨迹敏感性记录仍有效，本单对试验不提供稳定因果效应估计，未挑选多次重跑中的最好结果。

动态部分来自精确 I10_static_residual final2k（SHA `1a87d0da...f926d1a`），保留其全部非静态模型、完整动态/物理 Adam、RNG、相机剩余栈和物理图。静态部分仅来自用户已接受的 J4 joint_depth_trust final2k（SHA `41234718...7354b62`），保留其 4126 点及完整静态 Adam；未导入 J4 的动态/物理状态，未重复创建静态库。坐标、schema、静态几何逐值一致。统一动态池 15507 点是高斯容量，不是物体数量；身份仍未知。

本次使用 120 张 train 的完整 0.8 L1 + 0.2 DSSIM RGB，不沿用 I10 旧的背景目标替换，也没有新颜色/静动标签、物体数、GT 深度/法线或验证集选点。历史父模型有颜色条件掩膜的训练/选择来源，不能称全新无先验静态发现。Rubber Duck 是 PIDG 目标物理基准数据，并非本轮另采实拍。

静态 XYZ/scale/rotation 保持原 12k 状态，联合组仅透明度/SH 的 Adam 从 14k 到 16k；对照保持 14k。动态 canonical/grid/deform Adam 从 24k 到 26k。I10 速度/应力参数、Adam 800 步和图缓存精确不变；本轮是 RGB 联合续训，新增物理更新为零。保留原 epsilon 1e-15 与学习率，未改变拓扑。0.1 权重的 J4 父静态 D/alpha 仅是自参照，可能保留旧错误，不是独立几何真值。

原生 86/188 在 t2/t100/t200 的 RGB、静动分解、alpha、D/alpha 与六张原训练轨迹比较实际查看；另看两张动态 alpha、动态 D/alpha 和静态 D/alpha，以及两个解码接触图，共 30 个图文件。静态主体未见明显新增大缺失，但仍有边界晕、表面不平；c86/t200 的暗色椭圆支撑与拉丝、c188/t100 的碎裂及高 alpha 横向尾迹没有消失。六原轨迹视图不能代替同一时刻其他相机的独立 GT。

静态 alpha>0.5 像素在 86/188 从 134111/147546 变为 133871/147326；原支撑变缺失 242/220 像素。父支撑上的 D/alpha 平均变化约 0.000647/0.001003，仅是渲染代理。低 alpha 黑像素是缺失，不是零几何误差；静态 XYZ 固定和少量支撑变化都不证明正确几何。

11 测试、默认关闭 CLI、原生同值 RGB/alpha/D、预定梯度容差及 opacity-only 自深度路由通过。两组各 20 步短训练、每组两个独立进程的完整模型/双 Adam/绑定/RNG/相机/图精确恢复、同份原生梯度的 Adam 更新精确重放通过。旧 40 步 CUDA 轨迹失败保持原结论。首次补充 CLI 检查遗漏 PYTHONPATH 的环境失败记录保留，正确环境后通过，未改冻结训练代码。

219 个训练输入/源码与两个父权重冻结；评估代码/30 张 val 另行冻结。独立 CPU 审计核验两组起点、全 4000 相机采样、冻结参数/Adam/图、全 30 图指标；6 个完整 Gaussian PLY 的所有位置/SH/透明度/scale/rotation 字段逐值匹配，另 9 个世界 XYZ 诊断 PLY 在原生导出时匹配。180 图文件 SHA/解码有效，两 H264 Main/yuv420p/faststart 视频各完整解码 200 帧，接触图已查看；未实测浏览器播放。日志中的 J4_STEP/VIDEO 前缀来自复用冻结 J4 实现，实际输入、状态 stage 和产物均为独立 J6。

联合权重：`runs/pidg_staged_reconstruction/RubberDuck_I10_joint/v1/formal/joint_depth_trust/checkpoint.pt`，SHA `{ev['checkpoint_audits']['joint_depth_trust']['checkpoint_sha256']}`。视频与所有图片在 `output/staged_reconstruction/RubberDuck_I10_joint/v1/`；完整数值见 evaluation.json，审计见 independent_audit.json，人工记录见 manual_review.json。后续若继续，应另立针对动态支撑/跨相机时间泛化的候选；本轮不自动调权重、扫参或延长，现有证据也不足以断言 RGB 无解。
'''
    (report/'assessment.md').write_text(text,encoding='utf-8',newline='\n')
    (b/'progress.md').write_text('''# J6 progress

- d02b4976: fixed the human-authorized I10 non-static/J4 static composition and two-arm 2000-update protocol.
- 03e2a89a: 11 tests; 219 source/input hashes; native flags-off/gradient checks; two 20-update smoke arms; full two-Adam/RNG/sampler/graph independent-process exact recovery and fixed-native-gradient Adam replay passed.
- Both formal arms completed 2000 updates. All30 native val, 180 hashed image files, 6 full Gaussian + 9 XYZ PLY exports, two H264 videos each 200 decoded frames complete.
- Independent full-state/artifact audit passed; 30 image files actually inspected. PSNR 29.068226 initial / 31.401124 control / 31.994673 joint. Sampled static non-regression retained; dynamic alpha/depth cleanup/completeness failed.
- completed_not_promoted; pending:false. Human-requested training executed. No automatic extension, no old paused job resumed, no new physics updates. See v1/assessment.md and final_review.json.
''',encoding='utf-8',newline='\n')
    print('J6_FINALIZED',status,delta,flush=True)


if __name__=='__main__':main()
