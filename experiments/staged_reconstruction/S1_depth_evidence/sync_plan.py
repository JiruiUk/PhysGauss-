"""Append only this task's evidence/state to shared plans; preserve other diffs."""
import difflib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SECTION = '''

### S1-D0/D1：先修复深度证据再决定表面约束（2026-11-03）

独立只读审计固定原RGB推断后再读GT：原深度与同射线静态GT误差<=0.05仅
79.4%，<=0.02约52.6%；薄结构射线有深度56.8%，不能混同约99%空间点覆盖。
检查后深度导出法线平均误差30.8–30.9度；相同像素解析GT深度控制约0.04度，
GT理想量化到原192级网格已产生约16.2度，真实匹配MAE0.038仍高于理想量化
0.00814。暂不将当前深度/有限差分法线直接作为强监督或伪GT。

D0已提交be408aa4。D1固定候选协议在输出前预声明，7a411aa3提交实现及18测试；
默认关闭，保持全部候选/原盆地，连续逆深度半网格细化，4源拟合、另外2源
代价不退化才更新；其接受视角不是独立GT。单一两条件推断/独立评估及
原图复核已完成：深度MAE约0.0380→0.0368，法线30.8→26.7度，薄结构/P90略好；
约20.3%仍未在0.05内匹配静态GT。只保留局部数值改进，不重复D1、不宣称伪影
改善。下一步单候选审计源视角不确定性/深度区间，禁止GT选训练点；新的完整
Gaussian训练须另立协议及静态GT/原生恢复门槛。无全图强深度/法线或物理扩展。
见[深度证据评估](staged_reconstruction/S1_depth_evidence/assessment.md)与
[进度](staged_reconstruction/S1_depth_evidence/progress.md)。
'''
MAP = '''

## Optional RGB depth evidence refinement (2026-11-03)

`variants/staged_reconstruction/depth_refine/core.py` implements a default-off
bounded inverse-depth proposal and separate held-source acceptance; disabled
calls return the original depth tensor unchanged. `infer.py` preserves the
inherited candidate set and reads only RGB/calibration/time plus saved RGB-MVS
depths, with per-view save/stop and input/source locks. `evaluate.py` alone reads
independent static/depth/normal GT after inference, compares fixed candidates
and fixed original normal-valid pixels. No Gaussian optimizer, field, renderer,
loader or historical checkpoint is altered. `tests/test_depth_refine.py` checks
the vertex, bounds, invalid/zero steps, held-source refusal and flags-off path.
'''


def main():
    patches = []
    for name,section in [('experiments/adaptive_reconstruction_plan.md',SECTION),
            ('experiments/staged_reconstruction_plan.md',SECTION),('PIDG_REPO_MAP.md',MAP)]:
        base = subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8').replace('\r\n','\n')
        path = ROOT/name; working = path.read_text(encoding='utf-8').replace('\r\n','\n')
        assert section.strip() not in working
        path.write_text(working+section,encoding='utf-8',newline='\n')
        patches.extend(difflib.unified_diff(base.splitlines(True),(base+section).splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    (ROOT/'tmp/S1_depth_plans_cached.patch').write_text(''.join(patches),encoding='utf-8',newline='\n')
    print('DEPTH_PLANS_TASK_ONLY_PATCH_PREPARED')


if __name__ == '__main__': main()
