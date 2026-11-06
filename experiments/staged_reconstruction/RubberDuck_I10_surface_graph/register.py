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
        append([('PIDG_REPO_MAP.md',"""### L3 anchor-guided revocable local surface graph

- `variants/staged_reconstruction/i10_surface_graph/core.py`: fit-only bounded
  overlapping neighbor hypotheses, native-checked satellites, anchor spokes and
  triangle links, root-only measured RGB reprojection, local rotation-invariant
  length/strain constraints; default-off exact identity and unknown suspension.
- `runtime.py`: native full-scene compositing evidence, root dependency, per-node
  reversible confidence, seven scheduled refreshes, full controller/twoAdam/RNG/
  camera/physics state; pureRGB matched control has exactly zero auxiliary loss.
- `RubberDuck_I10_surface_graph/{launch,launch_trace,evaluate,audit,controller_audit}.py`:
  qualification/full-state recovery, matched training, immutable refresh models,
  independent artifact and native replay. Tests: test_i10_surface_graph.py.
- Satellite targets are parent-transported proxies, not new measured tracks or
  physical identities. No new static labels, count, depth/normal GT or physics.
""")]);return
    b=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_surface_graph/v1'
    f=read(b/'final_review.json');assert not f['pending_review'];m=read(b/'evaluation.json')['validation']['means']
    h=read(b/'hypotheses.json')['summary'];health=read(b/'initial_health.json')['summary']
    section=f'''### L3锚点引导的局部表面图：实际匹配训练完成

用户再次授权统一动态池/表面片/可撤销连接/局部形变；ec06e5ac先固定单候选。
两臂同J6完整模型/双Adam/RNG/相机起点，纯RGB与surface_graph各2000原生1600步，
静态/physics全冻。{h['anchors']}原RGB锚点，{h['neighbor_hypotheses']}重叠邻域假设，
{h['distinct_rows']}提议独立行；初态{health['active_neighbor_rows']}个合格邻域行、
{health['active_edges']}条活跃边。邻域目标为父投影相对偏移搬运代理，不是独立轨迹/GT；
只对原锚点施加重投影，图约束父时变距离，可暂停/撤销/恢复，无删点或物体数。
全30val：J6{m['J6_parent']['PSNR']:.6f}、纯RGB{m['rgb_control']['PSNR']:.6f}、
图约束{m['surface_graph']['PSNR']:.6f}dB。{read(b/'manual_review.json')['conclusion']}。
状态{f['status']}；完整两Adam/控制器恢复、全部7次原生复核重放、PLY/图/采样和
两段各200帧视频核验；原质量门槛保留，无颜色静态新参考/GT监督/物理更新。
详情RubberDuck_I10_surface_graph/v1/assessment.md，不自动延长/扫参。
'''
    append([(n,section) for n in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']])


if __name__=='__main__':main()
