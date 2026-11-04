"""Task-only cached patches; preserve concurrent main-plan/repo-map changes."""
import difflib
import subprocess
from variants.staged_reconstruction.prepare_data import ROOT

SECTION = '''### S1-J3进入完整Gaussian训练：固定父渲染深度自身参照（2026-11-04）

825d23a1预声明、c98f6381实施、2e3afbda资格完成：仅增加固定0.1的支持内
log1p(|D/alpha-parent_D/parent_alpha|)自身参照，teacher来自相同RGB父静态，
不是GT且可能保留错误支撑。仅该项连接static opacity梯度；联合全图RGB
照常更新静态SH和统一动态池，静态几何冻结。没有alpha复制/alpha-one/
新深度法线/DA-V2/GT训练输入、语义掩膜、物体数、删点、扫权重或新物理。

6新+6原测试、295训练输入/源码冻结通过。原生初始RGB/alpha/D精确，flags-off
返回原class/loss，初始RGB梯度原容差；隔离Lself只到opacity，固定h0.005差分
0.0002142100对autograd0.0002141777通过。两例各20步smoke及独立双进程
完整model/两个Adam/绑定/RNG/相机恢复与固定native梯度Adam更新精确通过。
用户授权验收沿用，旧40步CUDA轨迹和epsilon对照失败保留，没有重试或改判。

当前只训练diffuse/glossy两个新候选各6000步；匹配控制逐字节复用J1-A已完成
同起点/采样/完整Adam的frozen_static6000，不重复旧训练。原trainer/runtime/
renderer代码不变，进程内默认关适配器提供新增项。保留原epsilon1e-15/学习率/
初始化/采样和全部质量门槛，不凭smoke/自身参照当独立几何改善。

结束后独立评估全分母准确射线、GT静态RGB、表面/薄结构/动态泛化和外溢，
核验匹配6000采样/Adam/原生0/24/72及novel24所有时间、D/alpha/alpha、PLY
和完整解码视频并实际看图。当前没有新的改善结论或Rubber Duck86/188结果。
两条件全部原门槛通过人工复核后才另立真实协议。状态runs/.../S1_joint_depth_trust/v1，
报告experiments/staged_reconstruction/S1_joint_depth_trust；不重跑J1-A/J2或旧任务。'''

MAPPING = '''## Default-off native parent depth self-reference (S1-J3)

`variants/staged_reconstruction/joint_depth_trust/core.py` adds one robust
expected-depth self-reference on fixed parent-render support; disabling returns
the original JointScene/loss objects. `runtime.py` keeps the immutable RGB-derived
teacher and deterministic per-camera cache outside model/Adam state; the new
term connects only static opacity. The parent is not independent geometry GT.
`experiments/staged_reconstruction/S1_joint_depth_trust/launch.py` adapts the
frozen original trainer in-process, qualifies native finite differences and exact
full-state/fixed-gradient recovery, trains two candidates, and reuses pinned
historical matched controls without rerunning them. Independent evaluation/
export retain original quality gates. Tests: `tests/test_joint_depth_trust.py`.
No original PIDG code, physics or historical weights change; no promotion yet.'''


def main():
    patches = []
    for name, section in [('experiments/adaptive_reconstruction_plan.md', SECTION),
            ('experiments/staged_reconstruction_plan.md', SECTION), ('PIDG_REPO_MAP.md', MAPPING)]:
        head = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT).decode('utf-8')
        path = ROOT/name; current = path.read_text(encoding='utf-8')
        assert section.splitlines()[0] not in head and section.splitlines()[0] not in current
        updated = head.rstrip()+'\n\n\n'+section+'\n'
        path.write_text(current.rstrip()+'\n\n\n'+section+'\n', encoding='utf-8', newline='\n')
        patches.extend(difflib.unified_diff(head.splitlines(keepends=True), updated.splitlines(keepends=True),
            fromfile='a/'+name, tofile='b/'+name))
    path = ROOT/'runs/pidg_staged_reconstruction/S1_joint_depth_trust/v1/plans_cached.patch'
    path.write_text(''.join(patches), encoding='utf-8', newline='\n'); print(path)


if __name__ == '__main__': main()
