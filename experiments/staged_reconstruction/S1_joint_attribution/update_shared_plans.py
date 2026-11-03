"""Append only this milestone to the index, retaining unrelated worktree edits."""
import difflib
import subprocess
from pathlib import Path
from variants.staged_reconstruction.prepare_data import ROOT

SECTION = '''### S1-J2完成：联合清理主要损失是有效深度漂移，下一版需保护结构

18b323e0预声明、0a653f5a实现、882dac4f完整测量/独立复核。5测试、623文件
冻结，原32相机×两条件×opacity/SH四组合全量原生重放通过，c24两独立进程
四组RGB/alpha/D逐值一致，模型/几何无更新。实际看两例c0/c8/c24共12图。

novel8每例47782表面像素，lost3039/3458中2958/3386（97.33/97.92%）仍有效
但误差>0.05，只有81/72变缺失；gained947/1070，净−2092/−2388闭合原J1-A。
lost仍有效子集平均深度位移+0.01810/+0.02149；薄结构8417分母准确像素净增
123/171。不能把4–5百分点覆盖下降都叫破洞，也不能只用薄板局部增益升级。

SH单独改变使外溢下降38.67/36.83%，全部32相机alpha/D精确不变；固定参数
反事实的外溢对称边际份额opacity69.79/71.30%、SH30.21/28.70%，含交互，
不是训练因果或物理身份。RGB更干净而alpha仍有悬浮块/底座外晕，主体表面
深度漂移占主要损失。J2是0更新诊断；J1-A继续completed_not_promoted。

下一步先预声明一个保护opacity调整时有效深度的联合候选：可研究RGB父模型
原生渲染深度的自身参照，必须披露它不是GT、可能保护错误支撑，不用本次GT
图选训练像素，不使用失败D0–D7深度，不把SH变暗等同几何清理。先文献/原生
梯度可实现性与固定数值协议，再匹配两条件完整Gaussian及精确状态恢复验收；
原独立GT/结构门槛通过才另立Rubber Duck协议。不恢复旧暂停任务、不扫参。
报告：experiments/staged_reconstruction/S1_joint_attribution/v1/assessment.md。'''

MAPPING = '''## Frozen joint opacity/SH attribution (S1-J2)

`experiments/staged_reconstruction/S1_joint_attribution/audit.py` is an explicit,
default-off independent evaluator of four fixed opacity/SH interventions. It
reuses the unchanged native static renderer on pinned J1-A checkpoint values;
`core.py` closes all/novel static and thin ray-transition denominators and
symmetric marginal RGB contributions. `independent_review.py` re-renders all64
cameras without importing that accounting or mixer. Native SH-only alpha/depth
invariance and two-process exact output replay pass; no optimizer/model update,
GT training assets or identity estimates. Tests: `tests/test_joint_attribution.py`.
J1-A appearance improves but remains unqualified for real training after ray
accuracy loss. This maps a read-only reconstruction diagnostic, not new physics.'''


def main():
    patches = []
    for name, section in [('experiments/adaptive_reconstruction_plan.md', SECTION),
            ('experiments/staged_reconstruction_plan.md', SECTION), ('PIDG_REPO_MAP.md', MAPPING)]:
        head = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT).decode('utf-8')
        path = ROOT/name; worktree = path.read_text(encoding='utf-8')
        assert section.splitlines()[0] not in head and section.splitlines()[0] not in worktree
        final = head.rstrip()+'\n\n\n'+section+'\n'
        path.write_text(worktree.rstrip()+'\n\n\n'+section+'\n', encoding='utf-8', newline='\n')
        patches.extend(difflib.unified_diff(head.splitlines(keepends=True), final.splitlines(keepends=True),
            fromfile='a/'+name, tofile='b/'+name))
    patch = ROOT/'runs/pidg_staged_reconstruction/S1_joint_attribution/v1/plans_cached.patch'
    patch.write_text(''.join(patches), encoding='utf-8', newline='\n'); print(patch)


if __name__ == '__main__': main()
