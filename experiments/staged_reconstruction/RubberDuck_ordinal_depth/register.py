"""Task-only shared-document staging with unrelated edits preserved."""
import argparse
import difflib
import subprocess
from variants.staged_reconstruction.i10_ordinal_depth import runtime as a
from variants.staged_reconstruction.i10_tracking.io import ROOT,read


def append(name,section):
    path=ROOT/name;head=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8')
    current=path.read_text(encoding='utf-8');assert section.splitlines()[0] not in current
    path.write_text(current.rstrip()+'\n\n'+section,encoding='utf-8',newline='\n')
    new=head.rstrip()+'\n\n'+section
    patch=''.join(difflib.unified_diff(head.splitlines(True),new.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    target=a.RUNS/('cached_'+path.stem+'.patch');target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(patch,encoding='utf-8',newline='\n');subprocess.run(['git','apply','--cached','--whitespace=nowarn',str(target)],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['mapping','results']);args=p.parse_args()
    if args.stage=='mapping':
        append('PIDG_REPO_MAP.md','''### L8 explicitly authorized local ordinal-depth pilot

- `i10_ordinal_depth/core.py`: frozen Small robust order pairs with full/disjoint
  calibration intervals, fit/check separation, default-off one-sided robust log
  ranking and reversible evidence states; no absolute depth targets or labels.
- `runtime.py`: complete J6/twoAdam/RNG/sampler/physics restoration, native own-row
  visibility and RGB eligibility, local400 D/alpha supervision, pureRGB control,
  static/physics/topology frozen; checkpoints include evidence controller.
- `RubberDuck_ordinal_depth/launch.py`: native/flags-off/gradient qualification,
  accepted independent fixed-gradient Adam recovery, one matched600-step pair.
- Original training/config/Gaussian/deformation/velocity/stress/flow/momentum,
  renderer/loaders/metrics/checkpoints retain previously mapped PIDG paths;
  no physics extension or original PIDG rewrite. Test: test_i10_ordinal_depth.py.
''');return
    final=read(a.REPORT/'final_review.json');ev=read(a.REPORT/'evaluation.json')
    means=ev['validation']['means']
    section=f'''### L8 Rubber Duck局部相对深度训练试验

用户明确允许在L7全局覆盖不足后试训相对关系；旧T4/L7失败不改判。
固定Small及J6，保留RGB点假设与周围RGB三角化锚点的稳健前后顺序，
全/两份互斥fit标定区间不重叠才提议，无绝对深度损失、新颜色/静态标签/物体数/GT。
纯RGB与RGB+ordinal两臂各600步同完整J6/双Adam/RNG/相机起点，静态/physics/拓扑冻结。
全30val PSNR：J6 {means['J6_parent']['PSNR']:.6f}、RGB {means['rgb_control']['PSNR']:.6f}、
ordinal {means['ordinal_depth']['PSNR']:.6f}dB。状态{final['status']}，
保留J6，实际原生86/188早中晚及alpha/D/PLY/两视频审计见RubberDuck_ordinal_depth/v1/assessment.md。
局部排序或RGB收益不代表完整物体关系/准确几何/清理成功；未知及历史颜色来源保留。
'''
    for name in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']:append(name,section)


if __name__=='__main__':main()
