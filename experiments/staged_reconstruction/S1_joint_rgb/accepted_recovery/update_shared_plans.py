"""Cache task-owned J1 evidence without staging parallel shared-plan edits."""
import difflib
import subprocess
from pathlib import Path
from variants.staged_reconstruction.prepare_data import ROOT, read

BASE = Path(__file__).resolve().parent

PLAN = '''### S1-J1：联合RGB训练与经用户选定的完整恢复验收（2026-11-03）

J0仅证明梯度可达，见c9ebd762/4fbe1832；无更新，不是拖影消除。J1固定
全RGB统一canonical动态池，与静态全冻结/开放opacity+SH两臂；静态几何不动，
无新颜色掩膜、静态标签、物体数或GT监督。父模型是两例RGB派生S1-N control6k，
继承条件几何支持来源，不能称独立几何GT。每条件每臂计划6000新更新。

原f035486d协议/51b7c6f1实现通过6测试、275文件冻结、原生flags-off与梯度，
但连续40对20+20逐参数轨迹一致失败；原initial/RNG/相机状态精确。独立
即时加载及原生反传诊断发现断点前就有微小非确定性，不能混同于状态丢失。
ac16bc95/31f37abf固定共同epsilon1e-8对照也未过；不扫epsilon、保留失败。

用户明确选择“采用精确状态恢复验收（推荐）”，12aae457另行固定接受协议：
原模型、两个Adam及参数绑定、全部RNG/相机精确加载；独立原生RGB/alpha精确、
raw梯度按原1e-6/1e-5，使用同一份native梯度后完整Adam/model/constraint更新
精确一致。四个条件×臂全部通过，已与原40步失败分别存档，不能声称后者通过。
恢复原J1的epsilon1e-15及所有训练数值，未升级失败epsilon对照。286文件新冻结，
2个launcher测试与原6测试通过。恢复验收修订不改变任何重建质量/完整性门槛。

当前已进入串行完整Gaussian合成训练：两条件diffuse/glossy × frozen/revisable。
独立评估静态GT PSNR、可见/薄结构覆盖、RGB外溢、D/alpha全分母及动态泛化，
并导出固定训练0/24/72、新视角24/各时间、匹配PLY和全解码视频，实际查看后
再判断；此处无清理改善结论。正式状态见runs/.../S1_joint_rgb/accepted_recovery，
报告见S1_joint_rgb/accepted_recovery/v1。Rubber Duck仍是目标，合成通过后还须
另立真实匹配协议；尚无新86/188结果、不启动旧18k→24k，不恢复PAUSED任务。
原PIDG/历史权重、S3-U与未知身份及并行完整性工作保持，不新增本构/材料/力。'''

MAP = '''## Opt-in RGB-only joint Gaussian training (S1-J1 and accepted recovery)

`variants/staged_reconstruction/joint_rgb/runtime.py` uses the original PIDG
deformation/hash-grid and single depth-sorted union with a4000-row unknown-count
canonical dynamic bank. The only paired switch opens static opacity/DC/rest SH;
static geometry stays frozen. core is default-off; train saves both complete
Adam states, RNG, camera stack/visits/trace and immutable source/protocol binding.
Original40-step cross-run checks failed and stay archived; the fixed epsilon
control also failed. No source/protocol/failure was overwritten.

`S1_joint_rgb/accepted_recovery/launch.py` redirects paths only, leaving frozen
J1 training/model code unchanged. Human-approved acceptance checks exact full
serialization/bindings and same-native-gradient Adam update in two independent
processes for each of four case-arms, plus native RGB/alpha equality and raw
gradient tolerance. All four pass; it does not claim deterministic40-step native
training. Formal training uses original epsilon/rates and quality gates. A
serial runner independently evaluates/exports, then waits for manual image and
geometry review. No physics, original PIDG, renderer or old weight changes.'''


def main():
    assert read(BASE/'v1/recovery.json')['passed']
    assert read(ROOT/'runs/pidg_staged_reconstruction/S1_joint_rgb/accepted_recovery/pipeline.json')['status'] in ['training', 'evaluating', 'evaluated_pending_exports_and_manual_review']
    chunks = []
    for name, section in [('experiments/adaptive_reconstruction_plan.md', PLAN),
        ('experiments/staged_reconstruction_plan.md', PLAN), ('PIDG_REPO_MAP.md', MAP)]:
        head = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT).decode('utf-8')
        assert section.splitlines()[0] not in head
        final = head.rstrip()+'\n\n\n'+section+'\n'
        path = ROOT/name; current = path.read_text(encoding='utf-8')
        assert section.splitlines()[0] not in current
        path.write_text(current.rstrip()+'\n\n\n'+section+'\n', encoding='utf-8', newline='\n')
        chunks.extend(difflib.unified_diff(head.splitlines(keepends=True), final.splitlines(keepends=True),
            fromfile='a/'+name, tofile='b/'+name))
    patch = ROOT/'runs/pidg_staged_reconstruction/S1_joint_rgb/accepted_recovery/shared_plans_cached.patch'
    patch.write_text(''.join(chunks), encoding='utf-8', newline='\n'); print(patch)


if __name__ == '__main__': main()
