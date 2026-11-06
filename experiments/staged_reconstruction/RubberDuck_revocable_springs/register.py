"""Append only this task's sections to shared working files and index."""
import argparse
import difflib
from pathlib import Path
import subprocess
from variants.staged_reconstruction.i10_tracking.io import ROOT, read


def append(items):
    patches=[]
    for name,section in items:
        path=ROOT/name;head=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8')
        current=path.read_text(encoding='utf-8');assert section.splitlines()[0] not in current
        path.write_text(current.rstrip()+'\n\n'+section,encoding='utf-8',newline='\n')
        updated=head.rstrip()+'\n\n'+section
        patches.extend(difflib.unified_diff(head.splitlines(True),updated.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    patch=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_revocable_springs/cached.patch'
    patch.parent.mkdir(parents=True,exist_ok=True);patch.write_text(''.join(patches),encoding='utf-8',newline='\n')
    subprocess.run(['git','apply','--cached','--whitespace=nowarn',str(patch)],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['mapping','results']);mode=p.parse_args().mode
    if mode=='mapping':
        append([('PIDG_REPO_MAP.md','''### L6 revocable shared-stiffness spring geometry

- `variants/staged_reconstruction/i10_springs/core.py`: fixed positive shared k,
  fit-median parent rest references, robust normalized edge strain; default off.
- `structure.py`: immutable L3/L5 node/edge proposals and native qualification,
  same initial J6 bank replay, cached per-frame plus current-root guards,
  unknown pause/two-contradiction revoke/two-pass restore; fixed denominators.
- `runtime.py`: matched common L4 memory, spring-only candidate; complete state
  and unchanged frozen PIDG/static/physics. No mass/force/material inference.
- `RubberDuck_revocable_springs/{launch,evaluate,controller_audit,audit}.py`:
  native qualification, independent recovery and matched2000-step arms.
  Tests: `tests/test_i10_springs.py`; historical color lineage and early-biased
  coverage retained. Parent median rest is not independent material geometry GT.
''')]);return
    base=ROOT/'experiments/staged_reconstruction/RubberDuck_revocable_springs/v1'
    final=read(base/'final_review.json');assert not final['pending_review']
    m=read(base/'evaluation.json')['validation']['means']
    section=f'''### L6可撤销共享固定刚度弹簧：匹配试验完成

dcac0409预声明一个候选。两臂同完整J6及L4时序记忆各2000步原生1600；
候选只加共享固定k=1、总权重0.01的稳健归一化长度应变。
初始60片/275不同动态行/337片内边，固定参考为原6fit帧父长度中位数；
原生自身证据与可撤销资格保留，非真实材料静止长度，早期覆盖限制不变。
全30val PSNR：J6 {m['J6_parent']['PSNR']:.6f}、时序对照 {m['temporal_control']['PSNR']:.6f}、
弹簧候选 {m['revocable_spring']['PSNR']:.6f}dB。{read(base/'manual_review.json')['conclusion']}。
状态{final['status']}。没有新增颜色静态参考/语义/物体数量/GT或物理更新；
历史颜色来源、CUDA轨迹敏感性及旧失败保留。不自动延长/扫参。
详见RubberDuck_revocable_springs/v1/assessment.md。
'''
    append([(name,section) for name in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']])


if __name__=='__main__':main()
