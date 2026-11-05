"""Close the bounded T3/T4 evidence trial without bypassing admission."""
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha

B=Path(__file__).parent;V=B/'v1'


def main():
    for n in ['unit_tests.json','native_check.json','independent_replay.json','independent_accounting.json']:
        assert read(V/n)['passed']
    a=read(V/'audit.json');s=a['summary'];assert not s['bindings']['passed'] and not s['graph']['passed']
    images=read(V/'images.json');images.update(read(V/'diagrams.json'))
    for n,h in images.items():assert sha(ROOT/n)==h
    write(V/'manual_review.json',dict(actual_visual_review=True,images_sha256=images,
          findings=['All0/52/108 twelve-frame sheets and local-neighborhood/coverage diagrams actually reviewed.',
                    'Accepted observations cluster on small visible parts of duck appearance; neither full body nor hidden surface is verified.',
                    'Model-inferred neighbors remain near the displayed track on source views; this is not independent3D or physical identity evidence.',
                    'Rejected observations are unknown, not background labels; the current graph is early-heavy.'],
          new_reconstruction_or_artifact_reduction=False,independent_3D_GT=False,historical_color_parent_disclosed=True))
    (V/'assessment.md').write_text('''# T4：绑定修正有效，但邻域约束尚未获准进入训练

本轮完成用户要求中的绑定核验与邻域构图。没有进行新的Gaussian训练，尚未实测邻域损失带来的表面完整性提升。当前J6权重保持不变；原T2负结果、T3未过覆盖结果均保留。

T3对原123条绑定作交换拟合/验证帧与原生可见贡献检查：103通过交换、84通过可见性、73同时通过，70个持续ID，时间分布63/5/5。20条交换未通过、39条可见性未通过，交集9条。这不能证明RGB追踪错误，而是原“投影最近”选择不足以证明选中了实际参与成像的表面。

T4固定一个修正：从原222条RGB运动候选出发，先由原生前20 ID缓冲提名，再以完整静动深度排序的一热颜色渲染测量每个候选的实际合成贡献。6个fit帧和6个held帧独立提名/选择；held不替fit挑ID。仍用原RMS2px、全程4px、0.25px可见替代间隔、0.01世界/0.75px交换等价与至少8/12可见（含4/6held）。没有可见的分离替代只表示在有界已观测候选集合中唯一，不能证明绝对深度。没有读取颜色阈值静态参考、语义/实例标签、物体数、val/test或GT。

| 阶段 | 早 | 中 | 晚 | 总计 |
|---|---:|---:|---:|---:|
| 原RGB运动候选 |177|25|20|222|
| T4通过绑定核验 |105|8|9|122|
| 再通过持续局部邻域 |104|7|5|116|

共测量2678个轨迹/高斯候选对。122条通过绑定核验，涉及114个持续高斯ID；其中95条原先已准入、27条为新核验；87条保留原ID、35条改变为可见贡献支持的ID。它们仍是模型条件绑定，不是独立正确率或物理物体身份。

116条通过局部邻域检查，涉及108个锚点ID、677个邻居ID（邻居分母是15507个动态高斯，约4.37%；不代表表面面积，且与锚点可能重叠）。固定最近8、半径0.05、12帧始终接近、相对位移RMS<=0.02，至少3邻居；不按当前接触一帧的空间近邻直接连接。所有点/未知/被拒假设保留，未删点、未增长、未赋予物理身份。

原定每时间段至少10条的准入门槛未过：绑定8/9、邻域7/5均不足。覆盖不足已出现在原候选177/25/20，绑定修正不能单独解决它。实际图也显示支撑集中于少量可见纹理部位，不能把677邻居叫完整表面。因此未启动正式邻域训练，也没有新模型、PLY、86/188训练视频或伪影改善声明。

18测试、默认关闭CLI、原生RGB/alpha/D一致、一热打包/单独贡献一致、白色合成=alpha、完整模型/双Adam/绑定/RNG/相机余栈/physics-cache无更新检查通过。两独立进程逐值复现全部222来源、2678候选、原生贡献、选择与120帧父三维数组。另一个CPU评估器独立重算222轨迹的两split排名/代价/邻域及所有分母。0/52/108三套12帧图及邻域/覆盖图实际查看。父颜色条件历史明确保留；D与3D位置都是模型推断，未知/遮挡不叫动态或背景。

状态completed_not_admitted，optimizer_updates=0。下一机制应先核验跨窗口的持续关联：检查原始730条RGB一致轨迹中，静止兼容/未知短窗能否通过重叠观测与已核验运动轨迹接续。静止世界兼容不等于永久静态身份；该方向尚未执行，必须另立协议，不能直接计入本次122或降低门槛。只有资格通过，再进行软局部运动/形状约束的固定匹配训练和独立完整状态恢复。当前结果不足以断言RGB无解，也不声称追踪路线无效。
''',encoding='utf-8',newline='\n')
    status=dict(status='completed_not_admitted',pending=False,pending_review=False,optimizer_updates=0,
                reliable_bindings=122,qualified_neighbor_tracks=116,graph_by_band=[104,7,5],
                binding_gate_passed=False,neighborhood_gate_passed=False,
                new_training_model=False,artifact_or_completeness_improvement_measured=False,
                historical_color_parent_disclosed=True,automatic_extension=False)
    names=['audit.json','source_freeze.json','independent_replay.json','independent_accounting.json','native_check.json','unit_tests.json','manual_review.json','assessment.md','diagrams.json']
    write(V/'final_review.json',dict(**status,artifacts_sha256={n:sha(V/n) for n in names}))
    write(V/'final_status.json',status)
    write(ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_track_neighborhood/visible/v1/final_status.json',status)
    write(V/'final_validation.json',dict(passed=True,pending_review=False,all_required_review_complete=True,
          no_optimizer_or_new_model=True,new_evaluation_source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),B/'review.py']}))
    (B/'progress.md').write_text('# Progress\n\n- T4 complete:18 tests/native qualification/full independent replay/CPU accounting pass.222 original motion candidates,2678 candidate pairs;122 bindings/114 IDs (105/8/9),116 neighborhood tracks/108 anchors/677 neighbor IDs (104/7/5). Frozen10-per-third admission fails; zero updates, completed_not_admitted, pending:false. Five diagrams actually reviewed; no weight change or artifact/complete-surface improvement claim.\n',encoding='utf-8',newline='\n')
    base=B.parent/'progress.md';current=base.read_text(encoding='utf-8').rstrip()
    base.write_text(current+'\n- T4 visible-first correction completed with independent replay/accounting:122 bindings,116 neighborhood tracks,677 neighbor IDs;104/7/5 fails unchanged temporal admission. No training; see visible/v1/assessment.md and next_plan.md.\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
