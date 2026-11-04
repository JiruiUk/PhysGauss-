"""Report frozen J5 measurements without training or quality promotion."""
import argparse
from PIL import Image
from variants.staged_reconstruction.rubber_visibility import worker as w
from .qualification_repair import verify_supplement


def summarize():
    verify_supplement();data=w.read(w.REPORT/'audit.json');audit=w.read(w.REPORT/'independent_audit.json')
    assert audit['passed'] and audit['audit_sha256']==w.sha(w.REPORT/'audit.json')
    census=data['metadata'];rows=[]
    labels={'parent6k':'S3-U父模型','frozen_static':'J4静态冻结','joint_depth_trust':'J4静动联合'}
    for name,result in data['models'].items():
        for target,total in zip(result['targets'],result['source_totals']):
            rows.append('|'+ '|'.join([labels[name],f"{target['camera']}/{target['number']}",
                str(target['dark_high_alpha_pixels']),str(target['dynamic_alpha_gt05_pixels']),
                str(target['selected_canonical_rows']),f"{100*target['fraction_dark_alpha_explained_by_first20_ID_union']:.3f}",
                f"{total['weighted_observed_mean_RGB']:.4f}",f"{100*total['weight_on_observed_RGB_lt005']:.2f}",
                f"{total['visible_nonzero_sources']}/120"])+'|')
    body=f'''# J5 Rubber Duck 相机时间与暗色动态支撑审计

本轮为只读诊断：三个既有检查点，训练更新 **0**。未生成新模型、训练视频或PLY；
未读取val/test RGB，未提供新语义标签、静态/动态真值、数量或GT深度。

## 实际覆盖

train共有{census['frames']}张RGB、{census['unique_times']}个精确时间键、
{census['unique_camera_transforms']}个精确相机变换。同时间拥有多个不同相机的时间数为
**{census['same_time_multicamera_times']}**。每个相机的不同时间数范围为
{min(census['camera_time_counts'])}—{max(census['camera_time_counts'])}。
固定时间存在训练图像，也不能把别的时刻视图当作该时刻的多视角GT。

## 全分母与结果

诊断集合：原生1600动态单层alpha>0.5且clamp后RGB通道均值<0.05。
它只是固定评价子集，不是语义掩膜。选择该子集像素前20个原生贡献ID的并集；
随后用颜色覆盖渲染量出真实合成贡献。右侧是这批模型canonical行在全部120源
各自时刻、联合静动态深度排序下的图像支撑。六组允许重叠，不能相加成分区。

|检查点|相机/时间号|暗高alpha像素|全部高alpha像素|选中模型行|目标暗alpha解释%|源贡献加权观测RGB|源贡献落在暗观测%|非零源|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''+'\n'.join(rows)+'''

源RGB均值和暗观测比例的分母是该组在所有源相机-时间下的原生可见合成权重
总和；逐源分母与缺失状态见audit.json。暗观测不代表空场景，明亮观测不证明
正确几何，模型行跨时间对应不等于已知物理身份。没有固定跨时间相机独立GT。
“120/120非零源”是组级总贡献非零，不代表组内每个点在120帧都可见。
每个固定目标的完整像素分母为1600×900=1,440,000，不是只报选中子集。

J4联合模型的六组源贡献约75.88%—88.48%落在暗观测，仍有约11.52%—24.12%
落在较亮观测。各组自身定义/点集不同，表中像素数变化不能作匹配物体几何
准确率。86/200暗高alpha像素父17,788→冻结39,830→联合37,689，说明RGB收益
没有伴随这项预定诊断量下降到父水平；这仍不是独立GT伪影面积。

这批在新相机-时间组合中产生暗支撑的点，在部分已观测组合仍参与明亮区域的
成像。因此不能按暗色或单图残差直接删点。训练轨迹能解释RGB，不足以约束
离轨迹时的三维动态形状。J5没有证明唯一主因，也没有证明RGB无解。

## 测量与失败记录

- 原始5测试及校准补充测试通过；默认关闭不打开模型/数据。209个原输入/源码
  文件冻结，修复及独立评估源码另有补充冻结；三个父检查点SHA固定。
- 原qualification0因新诊断夹具主点用了w/2,h/2失败。保留原代码/冻结/日志，
  独立文件改成原生((w-1)/2,(h-1)/2)，不改变协议和门槛。2/3两进程逐值一致。
- 原生单高斯white RGB精确等于alpha；导出的weight缓冲区精确等于
  alpha*(1-alpha)，与真实alpha最大差0.569745。J5从未使用此权重。
  此限制已在uncertainty_motion审计识别，本轮独立核验而不修改渲染器。
- 原生RGB/alpha/D路径精确一致，三通道组打包与分开渲染通过；完整模型、
  两个Adam/绑定、RNG、相机采样和physics图缓存测量前后精确不变，无反传。
- 独立进程重算全部元数据、三个模型六固定目标及源索引0/24/72，目标分母、
  ID并集数量、原生颜色贡献及观测统计逐值一致。全部120源聚合分母与图像哈希
  独立复核通过。实际人工图像复核记录在manual_review.json。

## 来源披露和下一步

父S3-U历史曾使用颜色条件掩膜。J4和J5无新语义掩膜并不消除父来源条件。
沿用自身4126静态/15507未知数量统一动态池；没有预设物体数或确定身份。
Rubber Duck是PIDG物理基准场景，real目录标签不证明物理相机采集。

复用已完成uncertainty_motion的2195个跨时间RGB支持锚点证据。锚点是颜色支持
和运动图可行性，不是准确深度/完整表面或独立3D轨迹。并行“动态物体保持完整度”
任务正在实现reliable_motion时间变可靠锚点运动传播；本线不再重复同一机制。
下一联合机制是先让弱可见支撑受到可见锚点的运动约束，再评估与可撤销静态
opacity/SH联合的可行性；不能仅靠继续RGB拟合或压暗SH称几何清理成功。
需等该动态候选完成预检/匹配训练与原6×9、6×200完整度验收，读取实际证据。
通过后才另立本线静动联合完整状态匹配协议，未通过则不得直接拼接升级。
任何新训练仍需flags-off/原生梯度/完整双Adam与RNG相机独立精确恢复、保留
原结构和外观门槛。既有J4不重跑/延长，J0—J3/D0—D7/S1-C/N不重启。
'''
    (w.REPORT/'assessment.md').write_text(body,encoding='utf-8',newline='\n')
    outputs={}
    for cam in [86,188]:
        for number in [2,100,200]:
            sources=[w.OUT/f'{name}_c{cam}_t{number}_diagnostic.png' for name in data['models']]
            canvas=Image.new('RGB',(1600,259*3))
            for i,path in enumerate(sources):
                with Image.open(path) as image:canvas.paste(image,(0,259*i))
            target=w.OUT/f'comparison_c{cam}_t{number}.png';canvas.save(target)
            outputs[target.relative_to(w.ROOT).as_posix()]=w.sha(target)
    w.old.atomic_json(w.REPORT/'report_artifacts.json',{'files':outputs,'no_new_training':True,
        'source_sha256':w.sha(w.Path(__file__))})
    print('J5_REPORT_PENDING_ACTUAL_VISUAL_REVIEW',len(outputs),flush=True)


def complete():
    verify_supplement();ind=w.read(w.REPORT/'independent_audit.json');assert ind['passed']
    manual=w.read(w.REPORT/'manual_review.json');assert manual['actually_viewed']
    for item in manual['actually_viewed']:assert w.sha(w.ROOT/item['path'])==item['sha256']
    assert not (w.REPORT/'final_review.json').exists()
    record={'status':'completed_read_only_not_promoted','pending_review':False,'training_updates':0,
        'independent_metadata_targets_sources_and_denominators_exact':True,
        'all_three_complete_states_unchanged':True,'actual_native_diagnostic_images_reviewed':True,
        'new_geometry_quality_or_independent_GT_claim':False,'legacy_color_parent_disclosed':True,
        'failed_fixture_preserved':True,'next_mechanism':'Reuse parallel reliable_motion qualification and completeness results before separate static-dynamic joint protocol',
        'duplicate_training_started':False,'full_goal_achieved':False}
    w.old.atomic_json(w.REPORT/'final_review.json',record)
    w.old.atomic_json(w.RUNS/'final_status.json',record);w.old.atomic_json(w.RUNS/'state.json',record)
    w.old.atomic_json(w.RUNS/'pipeline.json',{'status':record['status'],'pending_review':False,
        'qualification':True,'formal_audit':True,'independent_replay':True,'manual_review':True})
    print('J5_COMPLETED_READ_ONLY_NOT_PROMOTED',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-rubber-visibility-audit',action='store_true')
    p.add_argument('--phase',choices=['summarize','complete'],default='summarize');args=p.parse_args()
    if not args.enable_rubber_visibility_audit:raise SystemExit('J5 report default off')
    if args.phase=='summarize':summarize()
    else:complete()


if __name__=='__main__':main()
