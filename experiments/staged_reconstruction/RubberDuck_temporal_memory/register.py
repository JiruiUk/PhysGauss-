"""Protect unrelated changes: stage only task additions to shared documents."""
import argparse
from pathlib import Path
import subprocess
import difflib
from variants.staged_reconstruction.i10_tracking.io import ROOT,read

def append(items):
    patch=[]
    for name,section in items:
        path=ROOT/name;head=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8')
        current=path.read_text(encoding='utf-8');assert section.splitlines()[0] not in current
        path.write_text(current.rstrip()+'\n\n'+section,encoding='utf-8',newline='\n')
        updated=head.rstrip()+'\n\n'+section
        patch.extend(difflib.unified_diff(head.splitlines(True),updated.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    path=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_temporal_memory/register_cached.patch'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(patch),encoding='utf-8',newline='\n')
    subprocess.run(['git','apply','--cached','--whitespace=nowarn',str(path)],cwd=ROOT,check=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['mapping','results']);args=parser.parse_args()
    if args.mode=='mapping':
        append([('PIDG_REPO_MAP.md','''### L4 RGB temporal surface memory and motion-compensated bundle fusion

- `variants/staged_reconstruction/i10_temporal_memory/core.py`: fit-only reciprocal
  raw RGB overlap stitching, unknown/conflict preservation, canonical association,
  actual current/distant-keyframe projection and motion residual; default off.
- `runtime.py`: explicit persistent observation bank/controller, native visibility
  and RGB checks, pause/revoke/recover, unchanged PIDG multi-time deformation and
  renderer, full model/twoAdam/RNG/camera/physics/memory checkpoints.
- `RubberDuck_temporal_memory/{launch,input_audit,controller_audit,evaluate,audit}.py`:
  unit/native qualification, exact state/fixed-gradient recovery, one matched
  pair, independent memory/native replay and artifact review. Tests:
  `tests/test_i10_temporal_memory.py`. RGB-derived hypotheses are not 3D GT;
  no object count, semantic/static labels, topology growth or physics additions.
''')]);return
    b=ROOT/'experiments/staged_reconstruction/RubberDuck_temporal_memory/v1'
    final=read(b/'final_review.json');assert not final['pending_review']
    ev=read(b/'evaluation.json')['validation']['means'];manual=read(b/'manual_review.json')
    text=f'''### L4跨帧表面记忆：一次匹配试验完成

aaf46932在输出前固定RGB片段拼接/前后关键帧联合优化协议。
730原追踪片段完整保留，75条跨窗口长记忆，22条通过原生可见性，
372个观测的时间三段356/16/0；这是局部16至20帧原型，未解决全时段表面融合。
同J6完整状态纯RGB与temporal_memory各2000步原生1600更新；静态/physics/拓扑冻结。
全30val PSNR：J6 {ev['J6_parent']['PSNR']:.6f}、对照 {ev['rgb_control']['PSNR']:.6f}、
记忆 {ev['temporal_memory']['PSNR']:.6f}dB。{manual['conclusion']}。
状态{final['status']}，完整恢复/原生刷新/PLY/两视频和实际图复核均完成；
无新颜色参考/物体数/独立几何GT/物理更新，不自动扫参或延长。
详见RubberDuck_temporal_memory/v1/assessment.md。
'''
    append([(p,text) for p in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']])

if __name__=='__main__':main()
