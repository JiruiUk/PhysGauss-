"""Task-only cached additions, preserving unrelated shared-file edits."""
import argparse
import difflib
import subprocess
from variants.staged_reconstruction.i10_tracking.io import ROOT,read


def append(items):
    patch=[]
    for name,section in items:
        path=ROOT/name;head=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8')
        current=path.read_text(encoding='utf-8');assert section.splitlines()[0] not in current
        path.write_text(current.rstrip()+'\n\n'+section,encoding='utf-8',newline='\n')
        updated=head.rstrip()+'\n\n'+section
        patch.extend(difflib.unified_diff(head.splitlines(True),updated.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    p=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_local_motion/register_cached.patch'
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(patch),encoding='utf-8',newline='\n')
    subprocess.run(['git','apply','--cached','--whitespace=nowarn',str(p)],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['mapping','results']);args=p.parse_args()
    if args.mode=='mapping':
        append([('PIDG_REPO_MAP.md','''### L1 supported-window I10 local motion training

- `variants/staged_reconstruction/i10_local_motion/core.py::support`: current-frame
  visible persistent observations and undirected pairs within audited12-frame windows;
  no graph extrapolation or new labels. `add_local`: robust relative-correction prior,
  default identity; parent-derived local offsets are not ground-truth geometry.
- `runtime.py::{load,objective,snapshot,install}`: isolated native PIDG adapter,
  exact J6 full-state start; static and physics frozen, dynamic Adam retained.
- `experiments/staged_reconstruction/RubberDuck_I10_local_motion/launch.py`:
  default-off preparation/native gradients, matched smoke/recovery/full training.
  Global T3/T4 coverage rejection remains; this is a separately scoped local trial.
- `tests/test_i10_local_motion.py`: nine locality/occlusion/shape/gradient tests,
  plus five retained direct-track tests. No constitutive/material/force additions.
''')]);return
    b=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_local_motion/v1'
    final=read(b/'final_review.json');assert not final['pending_review']
    ev=read(b/'evaluation.json');m=ev['validation']['means']
    section=f'''### I10局部有支撑时段实际训练完成（L1）

用户指出审计后未更新模型。保留T3/T4全时段覆盖未通过与0更新记录，另立397e3b35
局部训练协议，明确每段10条是全时段支撑条件，不能据此禁止一切受限局部实验。
使用同一J6完整模型/双Adam/RNG/相机/physics起点，静态和physics全冻；两组各2000
原生1600更新，均用完整RGB及同一可见审计轨迹，候选另加固定0.01局部相对修正。
1325观测覆盖89/120训练帧，9367去重邻接对覆盖71/120帧，其余仅RGB，无图外推。
无新颜色静态参考/语义/数量/GT，父颜色历史保留；局部父形状可能保留错误，非GT。

全30val平均PSNR：J6父{m['J6_parent']['PSNR']:.6f}，轨迹对照{m['audited_tracks']['PSNR']:.6f}，
局部运动{m['local_motion']['PSNR']:.6f}dB。实际人工结论：{read(b/'manual_review.json')['conclusion']}。
状态{final['status']}；14测试、原生flags-off/局部梯度与完整独立进程精确状态/
固定native梯度Adam恢复、最终采样/参数/PLY/视频审计通过。两视频各解码200帧；
同一对照仍有CUDA轨迹敏感性，拟合/父邻域一致/alpha/D不证明几何。未放宽RGB
或人工完整性门槛，未恢复旧PAUSED任务、未调参延长。细节见
RubberDuck_I10_local_motion/v1/assessment.md。
'''
    append([(n,section) for n in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']])


if __name__=='__main__':main()
