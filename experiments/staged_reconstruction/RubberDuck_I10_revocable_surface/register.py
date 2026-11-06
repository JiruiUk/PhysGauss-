"""Stage only task additions in shared files, preserving unrelated work."""
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
    path=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_revocable_surface/register_cached.patch'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(patch),encoding='utf-8',newline='\n')
    subprocess.run(['git','apply','--cached','--whitespace=nowarn',str(path)],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['mapping','results']);args=p.parse_args()
    if args.mode=='mapping':
        append([('PIDG_REPO_MAP.md','''### L2 revocable RGB surface-patch bindings and connections

- `variants/staged_reconstruction/i10_revocable_surface/core.py`: fit-only patch
  memberships/weights, independently bound patch connections, detached evidence,
  suspend/revoke/recover transitions, rotation-invariant length reference loss.
- `runtime.py::{measure,refresh,objective,snapshot,restore,train}`: native full-scene
  patch contributions and raw RGB NCC; scheduled checks; exact controller plus
  model/twoAdam/RNG/camera/physics state. Static/physics frozen; default off.
- `RubberDuck_I10_revocable_surface/{launch,launch_trace}.py`: isolated matched
  qualification/training; separately frozen I/O adapter saves model-only refresh
  evidence for replay without altering frozen training decisions or losses.
- `tests/test_i10_revocable_surface.py`: selection, unknown/revocation/recovery,
  held exclusion, overlap/window checks, rotations, gradients and disabled path.
- `evaluate.py`, `audit.py`, `controller_audit.py`: independent native RGB/artifact,
  complete refresh replay and explicit CPU decision accounting. No geometry GT,
  physical identities, new color static references or new physics are introduced.
''')]);return
    b=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_revocable_surface/v1'
    f=read(b/'final_review.json');assert not f['pending_review'];m=read(b/'evaluation.json')['validation']['means']
    section=f'''### L2可撤销表面绑定与局部连接：实际训练完成

用户授权执行；03499339预声明单候选。固定表面/可撤销表面两臂从相同J6完整状态
各2000原生1600更新，静态/physics全冻。122片（12多高斯片、129独立点），
fit仅提议3条独立表面片连接，初态2条通过；弱覆盖如实保留，无阈值或权重扫参。
候选每250步以原生可见贡献、投影和RGB纹理复核，暂停/撤销/恢复均为非学习证据
决策，held不接收附加轨迹/连接损失但仍有普通RGB，不是独立GT。
全30val PSNR：父{m['J6_parent']['PSNR']:.6f}、固定{m['fixed_surface']['PSNR']:.6f}、
可撤销{m['revocable_surface']['PSNR']:.6f}dB。{read(b/'manual_review.json')['conclusion']}。
状态{f['status']}；原生/状态恢复、全部7次刷新模型及证据重放、完整状态/PLY/图/
双视频200帧解码均核验。父颜色历史披露，无新颜色静态参考/语义/数量/GT；
无物理更新或正确三维身份声明，不以置信度/父长度一致性证明几何，不自动延长。
详见RubberDuck_I10_revocable_surface/v1/assessment.md。
'''
    append([(n,section) for n in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']])


if __name__=='__main__':main()
