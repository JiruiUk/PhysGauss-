"""Close only this fixed exploratory pilot after actual native/artifact review."""
import argparse
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha

B=ROOT/'experiments/staged_reconstruction/RubberDuck_relative_order'
R=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_relative_order/v1'


def main():
    ev=read(B/'v1/evaluation.json');manual=read(B/'v1/manual_review.json')
    assert manual['actual_images_viewed'] and manual['decoded_contacts_viewed']
    for name in ['independent_audit','pairs_audit','controller_audit','recovery','native_check']:
        assert read(B/f'v1/{name}.json')['passed'],name
    for receipt in ['preregistration.json','v1/source_freeze.json','v1/evaluation_freeze.json','v1/repair_freeze.json']:
        q=read(B/receipt)
        for name,h in q.get('files',q).items():assert sha(ROOT/name)==h,name
    p=read(B/'protocol.json');means=ev['validation']['means'];order=ev['ordinal_diagnostic']['models']
    usage=read(B/'v1/controller_audit.json')['arms'];status='completed_local_trial_not_promoted'
    final=dict(status=status,pending=False,pending_review=False,promoted=False,retained_model='J6',
        retained_parent_sha256=p['parent_sha256'],matched_completed_updates={a:600 for a in p['arms']},
        discarded_failed_attempt_updates=100,old_interval_failure_preserved=True,
        original_training_source_protocol_pairs_and_thresholds_unchanged=True,
        separate_metadata_refresh_repair=True,
        numeric_RGB_gates=dict(vs_control=ev['validation']['PSNR_nonregression_gate'],vs_parent=ev['validation']['parent_nonregression_gate']),
        visual_cleanup_pass=manual['cleanup_pass'],visual_completeness_pass=manual['completeness_pass'],
        full_state_and_artifact_audits_passed=True,coverage=dict(training_pairs=30,held_pairs=32,dynamic_rows=2,early_and_middle_pairs=0),
        absolute_depth_supervision=False,independent_geometry_GT=False,new_semantic_masks_counts=False,
        historical_color_conditioned_parent_lineage=True,old_global_gates_unchanged=True,
        automatic_extension=False,manual_review_sha256=sha(B/'v1/manual_review.json'),
        source_sha256=sha(Path(__file__)))
    write(B/'v1/final_review.json',final)
    for arm in p['arms']:
        checkpoint=read(R/'formal'/arm/'checkpoint.json');assert checkpoint['iteration']==600
        write(R/'formal'/arm/'final_status.json',dict(status=status,iteration=600,pending_review=False,
            checkpoint_sha256=checkpoint['sha256'],review_sha256=sha(B/'v1/final_review.json')))
    closed=dict(status=status,pending_review=False,pending=False,optimizer_running=False,retained_model='J6',
        completed_arms=p['arms'],review_sha256=sha(B/'v1/final_review.json'))
    write(R/'final_status.json',closed);write(R/'pipeline.json',closed);write(R/'active.json',closed)
    rows=[]
    for name in ['J6_parent','rgb_control','ordinal_depth']:
        f=order[name]['fit'];h=order[name]['held'];m=means[name]
        rows.append(f"| {name} | {m['PSNR']:.6f} | {m['SSIM']:.6f} | {f['order_correct']}/{f['denominator']} ({f['known']} known) | {h['order_correct']}/{h['denominator']} ({h['known']} known) |")
    text=f'''# L8-R Rubber Duck局部相对深度训练试验

完成匹配两臂各600步；状态 `{status}`，保留J6，不自动加训或晋级。
用户授权相对关系试训；原T4/L7全局覆盖、L8区间准入失败及历史权重均保留。

## 实际训练及关系来源

同一完整J6/两个Adam/RNG/剩余相机栈起点，原生1600 RGB；候选额外使用400分辨率
原生联合渲染D/alpha的单侧log顺序，weight=.001、margin=log1.02、scale=.05。
静态模型及Adam、physics/Adam/图缓存、拓扑不变，仅更新未知数量统一动态池。
未叠加时序/弹簧/绝对深度损失。原生flags-off RGB/alpha/D同值及原梯度容差通过；
原生D梯度有限，两独立进程完整状态恢复、同一原生梯度Adam更新精确通过。

Small raw-relative在持续RGB轨迹与静止相容RGB锚点之间提出顺序；不作尺度标定。
4个偶数帧支持同符号，至少4个奇数帧另查该符号；5x5范围/间隔检查、组隔离及每图4对上限固定。
全部7493个组假设及322个逐帧提议保留，固定30训练对/32诊断对，只涉及2个动态行。
训练仅索引100—114的8个偶数帧；诊断101—115的8个奇数帧；没有早/中期覆盖。
初始19对active、11对suspended。RGB/自身可见性及重投影资格每100步刷新，可撤销/恢复。
候选实际 {usage['ordinal_depth']['steps_using_pairs']} 步使用关系对，
{usage['ordinal_depth']['positive_auxiliary_steps']} 步产生非零辅助损失，
合计 {usage['ordinal_depth']['used_pair_observations']} 次有效pair观测。

## 数值结果

| 模型 | 全30val PSNR dB | SSIM | fit与先验顺序一致/固定分母 | held与先验顺序一致/固定分母 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

候选较纯RGB对照PSNR {means['ordinal_depth']['PSNR']-means['rgb_control']['PSNR']:+.6f}dB，
较J6 {means['ordinal_depth']['PSNR']-means['J6_parent']['PSNR']:+.6f}dB；
原两项最多下降0.3dB门槛：对照 {final['numeric_RGB_gates']['vs_control']}，父模型 {final['numeric_RGB_gates']['vs_parent']}。
所有122条原生轨迹/1464观测重投影另报evaluation.json；它们是训练RGB轨迹，不是独立GT。
held只避开辅助顺序训练，仍属于RGB训练图像；不是泛化几何真值。
本次fit一致数9/30、held一致数19/32三模型均相同（fit另有2对alpha不足未知）；
候选平均有向log间隔仅小幅移动，没有关系对翻转到正确先验顺序一侧。
相比对照，六固定图暗高alpha代理像素121627→113170，但仍高于J6的107498，
该统计不是真实几何面积，也不取代实际拖影/完整性复核。

## 实际图像及完整性

{manual['conclusion']}

实际查看固定86/188的2/100/200时刻原生图、RGB/alpha/D分解、观测轨迹图、两个完整解码视频接触图。
手工清理门槛 {manual['cleanup_pass']}，完整性门槛 {manual['completeness_pass']}。
静态RGB/alpha/D精确相同。6个完整Gaussian PLY全部字段匹配，9个世界XYZ诊断匹配；
两H264 Main/yuv420p/faststart各200帧独立解码，未声称浏览器播放已测。

## 审计与错误记录

20测试通过（含独立修复回归）。原生两臂各20步及恢复通过。
首个正式对照在100步后状态合并重复measurement而退出；无完整100步断点，未续接部分状态。
原失败日志/初始状态/进度保留于attempt1_failed_refresh；独立repair_run只过滤旧测量元数据，
原冻结代码、协议、损失/阈值/权重不变；重复恢复验收后两臂从原完整J6重新各600步。
因此保留比较共1200更新，另有失败对照100步及两轮短资格训练，不混为最终模型步数。
全部1200相机采样重放，双Adam/父状态/冻结physics与静态精确核验，10个已存刷新模型原生逐值重放，
独立原始patch区间/符号/组隔离/所有提议/上限算术核验通过，且全部图像/PLY/视频哈希核验。

## 解释边界

这里得到局部图像射线前后顺序假设，不知道语义上哪个行属于哪只鸭子，
锚点也只是静止相容RGB假设，不是静态标签；没有新颜色阈值/物体数量/GT深度法线输入。
父J6历史颜色条件来源披露，本次无新颜色分类不能抹去该来源。
原生D/alpha混合了同射线多个高斯的贡献，自身行准入并不等于该行独占深度；
排序或PSNR上升不能证明整物体距离、厚度、几何或伪影已正确。
低alpha/黑深度为缺失。Rubber Duck没有本试验可用的固定时间多视角独立几何GT。
下一步仅按本次实际覆盖与视觉证据决定，不自动扩大权重、扫参或替换J6。
若继续这一路线，优先解决跨时段可靠关系覆盖，以及联合D/alpha到具体表面节点的对应，
再讨论另一个固定试验；本次结果不支持简单加训或增加顺序损失权重。
'''
    (B/'v1/assessment.md').write_text(text,encoding='utf-8',newline='\n')
    (B/'progress.md').write_text('# Progress\n\n- Local recurring raw-order mechanism preregistered 5e1a0800; old interval failure retained.\n'
        '- Native/input qualification aa62478d; initial refresh failure preserved, independent metadata-only repair 6b2d6f3c, repaired exact recovery 4a8173d8.\n'
        '- Two complete matched600-step arms finished; all audits and actual native/video review closed, pending:false.\n'
        f'- {manual["conclusion"]}\n- J6 retained; assessment.md contains full fixed denominators and limits.\n',encoding='utf-8',newline='\n')
    print('L8R_CLOSED',status,means,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--enable-close',action='store_true')
    if not parser.parse_args().enable_close:raise SystemExit('Explicit closure required')
    main()
