"""Append only J8 and stage its independent cached patch, preserving other edits."""
import argparse
import difflib
import subprocess
from pathlib import Path
from variants.staged_reconstruction.prepare_data import ROOT,write,sha

B=ROOT/'experiments/staged_reconstruction/RubberDuck_J7_long_memory'
MARKER='J8 J7 observed long-term memory'
PLAN='''

### J8 J7 observed long-term memory：匹配训练及复核完成

用户要求J7加入长期记忆，已完成同一J7完整模型/双Adam/RNG/相机起点的两臂各2000步。
保留无蓝色阈值3273静态库、15507统一动态池及历史动态颜色来源；两臂均训练静态opacity/SH
及动态，冻结静态几何/physics/拓扑，保留自身J7深度教师。候选只加冻结L4观测记忆0.01。
全部730轨迹/641记忆保留，J7重新计算120时刻全轨迹并复核：21绑定、17初始活跃，
292观测三段276/16/0；1750步末次复核剩8活跃/140前段观测，12暂停、1撤销。
203/2000步实际加记忆、3634端点对，未覆盖全时段/全表面，不是Transformer/RNN或新增表面。
全30val PSNR：J7 34.045075、匹配普通续训34.300525、记忆34.173143dB；
记忆对照−0.127383、父+0.128067dB，两项0.3dB门槛通过。21原绑定共360观测，
fit/held重投影对照1.026963/1.026156→0.652001/0.650365px，held仍受普通RGB训练、非GT。
实际86/188早中晚原图/分解/轨迹/解码图复核：静态RGB主体保持，动态碎裂和暗高alpha团块/尖尾仍在，
86晚期暗高alpha57522对46159反增24.62%；这是渲染代理，不按暗色删点/判背景。
18测试、原生flags-off/梯度、各20短训及四进程完整精确双Adam/RNG/相机/控制恢复通过；
532训练/35评估输入源码冻结、4000采样、七次记忆刷新、180PNG、6完整Gaussian+9XYZ、
两H264各200帧独立解码审计完成。保留局部时序收益，不升级、不重跑/延长/扫参；
completed_not_promoted，pending:false。原J7及旧40步/L4失败保留，无新增物理更新。
下一研究先固定更广可靠表面记忆与重现覆盖验收，不自动放大同一小组轨迹权重或开新候选。
详见experiments/staged_reconstruction/RubberDuck_J7_long_memory/v1/assessment.md。
'''
MAPPING='''

### J8 J7 observed long-term memory

- `variants/staged_reconstruction/j7_long_memory/runtime.py`: opt-in full J7/twoAdam/
  RNG/camera/physics restoration, own static teacher, static opacity/SH plus dynamic
  updates, persistent observation-memory controller and exact accepted recovery.
- Reuses frozen `i10_temporal_memory/core.py` stitching/binding/measurement/loss and
  its runtime native visibility/refresh without editing them. Old row nominations
  are rechecked on all120 freshly computed J7 trajectories; old xyz/admissions are not reused.
- `RubberDuck_J7_long_memory/launch.py`: 18 tests, fresh bank/native admission,
  flags-off/gradient/exception checks, two20 smokes/four recoveries and serial2000 pairs.
- `input_audit.py`, `trajectory_audit.py`, `audit.py`, `controller_audit.py`: independent
  retained-track/held isolation, parent trajectories, all4000 samples/twoAdam/frozen
  physics, seven native refresh and controller replay. `evaluate.py`/`diagnostics.py`
  compare J7/control/memory on native RGB/alpha/D, all30val, fixed86/188, PLY and video.
- `final_verify.py` audits all full-Gaussian fields, means, images/hashes and200-frame
  decoding; `finalize.py` closes actual review without automatic promotion. Existing
  PIDG training/render/physics/data/checkpoint sources remain unchanged. Dynamic
  historical color lineage is disclosed; persistent memory is not object identity,
  full-sequence coverage, new surface generation or independently measured3D.
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--apply-task-docs',action='store_true')
    if not p.parse_args().apply_task_docs:raise SystemExit('Explicit task documentation update required')
    assert read_review()['pending'] is False
    patches=[];receipt={}
    for name,addition in {'PIDG_REPO_MAP.md':MAPPING,'experiments/adaptive_reconstruction_plan.md':PLAN,'experiments/staged_reconstruction_plan.md':PLAN}.items():
        path=ROOT/name;assert MARKER not in path.read_text(encoding='utf-8-sig')
        previous=sha(path);original=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8')
        assert original.endswith('\n')
        patches.extend(difflib.unified_diff(original.splitlines(keepends=True),(original+addition).splitlines(keepends=True),fromfile='a/'+name,tofile='b/'+name))
        with path.open('ab') as f:f.write(addition.encode('utf-8'))
        receipt[name]=dict(previous_worktree_sha256=previous,appended_worktree_sha256=sha(path),previous_worktree_bytes_preserved=True,cached_patch_only_J8=True)
    path=B/'v1/shared_cached.patch';path.write_text(''.join(patches),encoding='utf-8',newline='\n')
    write(B/'v1/shared_append_receipt.json',dict(files=receipt,patch_sha256=sha(path),source_sha256=sha(Path(__file__))))

def read_review():
    from variants.staged_reconstruction.prepare_data import read
    return read(B/'v1/final_review.json')

if __name__=='__main__':main()
