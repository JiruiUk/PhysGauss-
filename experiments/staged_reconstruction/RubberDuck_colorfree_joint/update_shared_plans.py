"""Append only J7 documentation; make a cached-only patch against HEAD."""
import argparse
import difflib
import subprocess
from variants.staged_reconstruction.prepare_data import ROOT, write, sha

B=ROOT/'experiments/staged_reconstruction/RubberDuck_colorfree_joint'
MARKER='J7 color-free static library replacement'
PLAN='''

### J7 color-free static library replacement：融合匹配训练完成

用户要求先展示无蓝色阈值静态重建、再融合J6实测，现已完整执行。
3273点S1-G/S1-C control18k替换J6的4126点静态；两组相同J6完整动态/Adam/RNG/
相机起点，各2000原生1600更新。两组均训练静态opacity/SH及动态，静态几何和physics冻结；
候选用自身无颜色静态库作深度自参照，不用旧蓝色条件库深度。没有新颜色掩膜/数量/GT。
动态历史颜色来源、静态点数/Adam/教师差别披露；不是只切换颜色规则的单因素消融。
全30val PSNR：J6 31.994673、直接替换29.393544、旧库对照32.106181、候选34.045075dB；
候选较对照+1.938895dB，较J6+2.050402dB，RGB门槛通过。实际35图复核：静态RGB残留减弱，
主体仍在，但静态alpha广支撑/尖刺和动态碎裂/暗高alpha拖尾仍在，完整性与清理门槛失败。
9测试、原生flags-off/梯度、两组20步和各两进程完整双Adam/RNG/相机精确恢复通过；
505训练源/输入、35评估源/输入、4000采样、234图、8完整Gaussian+12XYZ PLY审计，
两H264各200帧独立全解码通过。保留外观收益和原J6，不自动替换/延长/扫参；
status=completed_rgb_improved_not_geometry_promoted，pending:false，无新增物理更新。
详见experiments/staged_reconstruction/RubberDuck_colorfree_joint/v1/assessment.md。
'''
MAPPING='''

### J7 color-free static library replacement

- `variants/staged_reconstruction/colorfree_joint/core.py`: schema/finite/row-ID
  validation and explicit static-module replacement; default-off returns caller unchanged.
- `runtime.py`: exact J6 non-static/two-Adam/RNG/camera/physics graph composition
  with selected3273-row RGB-geometric static bank; each arm's own static teacher,
  opacity/SH and dynamic-only updates, frozen static geometry/physics, full snapshots.
- `RubberDuck_colorfree_joint/launch.py`: opt-in qualification and one serial matched
  2000-step pair. `audit.py` replays source/state/Adam/sampling independently;
  `evaluate.py`/`diagnostics.py` compare4models using native PIDG RGB/alpha/D,
  all30val, fixed86/188 and original trajectory, PLY and decoded H264.
- `final_verify.py` verifies all full-Gaussian fields, frozen inputs, images/means
  and complete video decoding. `finalize.py` records actual reviewed result without promotion.
- Tests: `tests/test_colorfree_joint.py`. Original PIDG training/config/Gaussian/
  deformation/velocity/stress/flow/momentum/render/data/checkpoint paths above remain
  unchanged. No new physical field, object count or semantic labels. Dynamic historic
  color lineage persists; source replacement is not a from-scratch label-free model.
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--apply-task-docs',action='store_true')
    if not p.parse_args().apply_task_docs:raise SystemExit('Explicit documentation update required')
    changes={'PIDG_REPO_MAP.md':MAPPING,'experiments/adaptive_reconstruction_plan.md':PLAN,
             'experiments/staged_reconstruction_plan.md':PLAN}
    patches=[];receipt={}
    for name,addition in changes.items():
        path=ROOT/name;assert MARKER not in path.read_text(encoding='utf-8-sig')
        before=sha(path)
        original=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8')
        assert original.endswith('\n')
        patches.extend(difflib.unified_diff(original.splitlines(keepends=True),
            (original+addition).splitlines(keepends=True),fromfile='a/'+name,tofile='b/'+name))
        with path.open('ab') as stream:stream.write(addition.encode('utf-8'))
        receipt[name]=dict(previous_worktree_sha256=before,appended_worktree_sha256=sha(path),
                           previous_worktree_bytes_preserved=True,cached_patch_contains_only_J7=True)
    patch=B/'v1/shared_cached.patch';patch.write_text(''.join(patches),encoding='utf-8',newline='\n')
    write(B/'v1/shared_append_receipt.json',dict(files=receipt,patch_sha256=sha(patch),source_sha256=sha(__file__)))
    print('J7_WORKTREE_APPENDED_CACHED_PATCH_PREPARED',flush=True)


if __name__=='__main__':main()
