"""Task-only shared documentation hunks; preserve unrelated working changes."""
import argparse
import difflib
import subprocess
from variants.staged_reconstruction.i10_pseudo_depth import io


def append(name,section):
    path=io.ROOT/name;head=subprocess.check_output(['git','show','HEAD:'+name],cwd=io.ROOT).decode('utf-8')
    current=path.read_text(encoding='utf-8');assert section.splitlines()[0] not in current
    path.write_text(current.rstrip()+'\n\n'+section,encoding='utf-8',newline='\n')
    new=head.rstrip()+'\n\n'+section
    patch=''.join(difflib.unified_diff(head.splitlines(True),new.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    target=io.RUNS/('cached_'+path.stem+'.patch');target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(patch,encoding='utf-8',newline='\n')
    subprocess.run(['git','apply','--cached','--whitespace=nowarn',str(target)],cwd=io.ROOT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['mapping','results']);args=p.parse_args()
    if args.stage=='mapping':
        append('PIDG_REPO_MAP.md','''### L7-P Rubber Duck pseudo-depth admission

- `i10_pseudo_depth/core.py`: spatially separated RGB triangulation calibration,
  no-extrapolation hull, camera/world-motion compensation, discontinuity proxies,
  detached robust weak log-depth loss and revoke/recover state; default off.
- `io.py`: task-local lock/stop and frozen source/weights/train inputs; unchanged
  PIDG target compositing without using image alpha as an evidence selector.
- `audit.py`: all120 calibrations, existing T4 motion binding checks and all4433
  raw tracker-proposal boundaries; all rejected/missing observations retained.
- `RubberDuck_pseudo_depth/run.py`: inference, full CPU/fixed prior independent
  process replay; `review.py` fixed raw RGB panels. Tests: test_i10_pseudo_depth.py.
- Training entry/config/render/checkpoint are unchanged PIDG/J6 paths above;
  this stage has zero optimizer updates. Native training requires a new matched
  protocol after its local evidence gate, not promotion from proxy agreement.
''');return
    final=io.read(io.REPORT/'final_review.json');a=io.read(io.REPORT/'audit.json')['summary']
    section=f'''### L7-P Rubber Duck伪深度准入审计

固定DA-V2 Small，全部120训练RGB；尺度由RGB静止相容三角化锚点分空间拟合/检查，
不以J6渲染深度标定，不将其当静态标签或GT。全部4433追踪提议保留。
标定通过{a['calibrated_views']}/120，三时段{a['calibrated_views_by_third']}；
运动补偿及局部范围/边界检查后{a['reliable_motion_tracks']}条，三时段{a['reliable_tracks_by_third']}，
{a['distinct_rows']}个不同高斯行。局部准入{a['local_evidence_gate_passed']}。
状态{final['status']}；本阶段0次Gaussian更新，保留J6、旧全局覆盖失败和历史颜色来源。
详见RubberDuck_pseudo_depth/v1/assessment.md。未知不当背景/动态，不放宽门槛。
'''
    for name in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']:append(name,section)


if __name__=='__main__':main()
