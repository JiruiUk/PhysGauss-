"""Only task-specific cached hunks; unrelated working changes are retained."""
import argparse
import difflib
import subprocess
from variants.staged_reconstruction.i10_joint_scale import io


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
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['mapping','results']);args=parser.parse_args()
    if args.stage=='mapping':
        append('PIDG_REPO_MAP.md','''### L7-J Rubber Duck cross-frame scale and wider RGB anchors

- `i10_joint_scale/core.py`: full-image tiled Shi-Tomasi/LK bidirectional texture
  tracks, stationary RGB triangulation hypotheses, persistent duplicate groups,
  group-held calibration, positive normalized affine temporal fit; default off.
- `io.py`: own lock/stop, frozen L7-P raw priors/J6/RGB/code; no native mutations.
- `audit.py`: same-expanded-anchor framewise diagnostic control, fixed local
  calibration gates, unchanged T4 motion and full4433x12 boundary denominators.
- `RubberDuck_joint_scale/run.py`: freeze/extract/measure/exact independent replay;
  tests: `test_i10_joint_scale.py`. No optimizer/renderer/checkpoint code changes.
  Original PIDG training, Gaussian/deformation/velocity/stress/flow/momentum,
  loaders, metrics and checkpoint paths remain as mapped above. This audit is
  not a new physics milestone or Gaussian training qualification.
''');return
    a=io.read(io.REPORT/'audit.json');s=a['candidate']['summary'];control=a['framewise_control']['summary']
    text=f'''### L7-J Rubber Duck跨帧联合尺度及更广RGB锚点审计

固定Small/L7-P原始输出及J6。新增全图纹理双向LK，只将原几何门槛通过的静止相容
轨迹作尺度假设；无新颜色/静态标签/数量/GT。整组fit/check隔离，局部4px去重，
联合120帧正斜率仿射标定，原局部空间/数值/动态准入不放宽。
原静止相容{a['original_stationary_tracks']}条，新增{a['expanded_stationary_tracks']}条；
联合标定{s['calibrated_views']}/120，早中晚{s['calibrated_views_by_third']}，
同扩展锚点逐帧诊断对照{control['calibrated_views']}/120。
运动数值检查到达{s['motion_numeric_test_reached']}条，合格{s['reliable_motion_tracks']}条、
{s['distinct_rows']}个不同高斯行；局部准入{s['local_evidence_gate_passed']}。
0次Gaussian更新，J6保留；原T4全局失败、历史颜色来源和未知身份不改判。
详见RubberDuck_joint_scale/v1/assessment.md。
'''
    for name in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']:append(name,text)


if __name__=='__main__':main()
