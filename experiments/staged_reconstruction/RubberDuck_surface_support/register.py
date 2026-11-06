"""Stage task-only appended sections while preserving shared working changes."""
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
    patch=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_surface_support/cached.patch'
    patch.parent.mkdir(parents=True,exist_ok=True);patch.write_text(''.join(patches),encoding='utf-8',newline='\n')
    subprocess.run(['git','apply','--cached','--whitespace=nowarn',str(patch)],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['mapping','results']);mode=p.parse_args().mode
    if mode=='mapping':
        append([('PIDG_REPO_MAP.md','''### L5 observed local surface support

- `variants/staged_reconstruction/i10_surface_support/core.py`: full Gaussian
  covariance, rotation-invariant support spectrum/area intervals, ellipsoid gap
  and noncollapse, actual observed bilinear footprint contribution; default off.
- `structure.py`: reuse immutable L3 hypotheses/native gates, fit-only J6
  reference, revocable own-node support, bounded native group contribution.
- `runtime.py`: common L4 temporal supervision, candidate-only local structure,
  complete twoAdam/RNG/camera/physics/combined-controller state, baseline intact.
- `RubberDuck_surface_support/{launch,evaluate,audit,controller_audit,
  structural_diagnostic}.py`: matched native qualification/training and fixed
  denominator evaluation; tests `tests/test_i10_surface_support.py`.
  Local support thickness is not true body thickness or independent geometry GT.
''')]);return
    base=ROOT/'experiments/staged_reconstruction/RubberDuck_surface_support/v1'
    final=read(base/'final_review.json');assert not final['pending_review']
    ev=read(base/'evaluation.json')['validation']['means'];summary=read(base/'surface_bank.json')['summary']
    section=f'''### L5连接、局部支撑厚度与观测完整性：匹配试验完成

3ca738e1先固定一个候选。两臂同J6完整起点及L4时序损失，各2000步原生1600；
候选增加椭球连接间隙/防塌缩、局部支撑厚度区间、切向面积及单点尺度上界、
原生观测轨迹足迹贡献保持，总结构权重0.01。无GT/新颜色静态参考/数量或物理更新。
初态{summary['local_patches']}局部片/{summary['rows']}行/{summary['edges']}边，时间三段{summary['temporal_thirds']}；
这是父模型条件支撑先验，不能当真实体厚、全时段完整性或正确几何证明。
全30val PSNR：J6 {ev['J6_parent']['PSNR']:.6f}、时序对照 {ev['temporal_control']['PSNR']:.6f}、
结构候选 {ev['surface_support']['PSNR']:.6f}dB。{read(base/'manual_review.json')['conclusion']}。
状态{final['status']}，原质量门槛保留；不自动延长/扫参。
详见RubberDuck_surface_support/v1/assessment.md。
'''
    append([(name,section) for name in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']])


if __name__=='__main__':main()
