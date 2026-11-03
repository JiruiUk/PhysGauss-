"""Append J0 evidence and stage only task-owned sections, preserving shared edits."""
import difflib
import subprocess
from variants.staged_reconstruction.prepare_data import ROOT

PLAN = '''### S1-J0：静态撤销梯度资格通过，尚无更新/清理结果（2026-11-03）

24ccb92d协议，b7e304ba实现与预检，c9ebd762六固定训练帧测量/独立复核完成。
全图RGB原生单次深度排序使静态opacity/SH和动态canonical/grid/deform六组
都有有限非零梯度；两独立进程800/1600 RGB/alpha与参数精确一致，原梯度
atol1e-6/rtol1e-5通过，flags/叶梯度/值/RNG不变。9+2测试、391文件冻结、
54图哈希解码与原checkpoint核验通过。两次失败及修复receipt单独保留；冻结
worker、门槛、权重不改。六图opacity正/负梯度不能当静动身份或删点证据。

实际看86/188早中晚分解及两张1600原图，仍见动态碎裂/拉长、静态边缘模糊。
这是旧S3-U及其自身4126点条件静态库，历史曾用颜色掩膜；本次全RGB诊断
没有重训未知静态，未拼接新S1-C。optimizer更新0，未验证完整Adam训练恢复，
未证明清理/几何正确；final_review/status已完成，无新训练PLY/视频。

下一步另立一个匹配RGB联合Gaussian实验：统一canonical动态池，比较静态
全冻结与开放opacity/SH，保持静态几何不动。先固定两条件合成协议及输入
来源，完成原生flags-off/梯度、独立完整Adam/RNG/相机恢复及独立静态GT/
结构/外溢门槛，才考虑Rubber Duck。失败D7深度/法线不进监督，原门槛不放宽，
不恢复暂停任务，不改并行完整性训练。见S1_joint_revocation_audit/v1/assessment.md。'''

MAP = '''## No-update native static-revocation audit (S1-J0, 2026-11-03)

`joint_revocation_audit/core.py` temporarily enables static opacity/SH autograd
and restores flags/values/leaf gradients/RNG, including exceptions. worker uses
the pinned historical S3-U scene and unchanged native renderer/full RGB loss;
no optimizer or sampler updates. Independent receipt wrapper handles virtual
Torch source paths without editing the frozen worker. preflight reloads in two
processes; export_review checks six bindings/hashes/sign denominators/54 images.
finalize records manual old-native-image review and clears pending state under
both locks. 9 core+2 repair tests pass, gradient path qualified; geometry and
cleanup unproven. This is not newly trained unknown-static reconstruction.
All original PIDG/weights/paused stages unchanged. Reports: S1_joint_revocation_audit/v1.'''


def main():
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
    path = ROOT/'runs/pidg_staged_reconstruction/S1_joint_revocation_audit/v1/shared_plans_cached.patch'
    path.write_text(''.join(chunks), encoding='utf-8', newline='\n')
    print(path)


if __name__ == '__main__':
    main()
