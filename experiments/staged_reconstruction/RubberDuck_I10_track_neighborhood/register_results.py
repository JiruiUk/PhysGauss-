"""Append only this task's finished evidence to shared maps/plans in the index."""
import difflib
from pathlib import Path
import subprocess
from variants.staged_reconstruction.i10_tracking.io import ROOT,read


def main():
    base=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_track_neighborhood'
    assert not read(base/'visible/v1/final_review.json')['pending_review']
    section='''### I10绑定可靠性与持续局部邻域核验（T3/T4，2026-11-05）

用户要求先核验绑定，再用可靠轨迹约束邻域共同运动/表面完整性。T3原123绑定
103通过交换fit/held、84通过原生实际合成贡献、73同时通过，70ID、63/5/5，
未过既定每段10条。T4改为可见贡献优先提名，保留原RGB轨迹、数值门槛及父权重：
222运动候选（177/25/20）、2678轨迹/高斯对，122绑定/114ID（105/8/9）；
其中27条新核验、35条更换原ID，不是独立绑定正确率。原生前20 ID只提名，
使用完整静动排序的一热颜色测量实际贡献，不把post-T缓冲当权重；fit与held
各自独立提名/选择。18测试、原生flags-off/打包资格、全222/120帧独立逐值重放
及CPU分母/排名/代价/邻域审计通过，完整模型/双Adam/RNG/相机余栈/physics不变。

116条具备持续局部邻域，108锚点ID/677邻居ID，但104/7/5仍未过每段10门槛；
实际固定0/52/108图及邻域图显示支撑局部集中，不能称完整表面。两阶段均为
completed_not_admitted/pending:false，0更新，无新训练模型/视频/PLY或伪影改善。
不放宽门槛、不恢复PAUSED任务；没有新颜色静态参考/标签/数量或GT，父颜色历史
仍披露。未知/遮挡不叫背景/动态，三维位置和邻域只是模型条件推断。
详见RubberDuck_I10_track_neighborhood/visible/v1/assessment.md。下一独立机制
可检验原730一致轨迹的跨窗口接续，不能把短窗静止兼容解释等同永久静态身份；
尚未执行或计入本轮。资格通过后才另立软局部运动/形状的匹配训练协议。
'''
    mapping='''### Audited I10 binding and persistent local-neighborhood evidence (T3/T4)

- `variants/staged_reconstruction/i10_track_neighborhood/core.py`: reciprocal
  split/equivalent-surface checks, native patch visibility and short-trajectory
  local neighbors; model-conditioned, no physical instances or GT.
- `audit.py`: opt-in123-binding native visibility/no-mutation audit, independent replay.
- `visible_core.py` and `visible.py`: split-independent first20 native-ID nomination,
  actual one-hot compositing, persistent visible-first binding over222 original tracks.
- `experiments/staged_reconstruction/RubberDuck_I10_track_neighborhood/`: fixed
  protocols,337/343-file freezes, independent accounting, real source-RGB diagrams.
  T3/T4 fail temporal coverage; zero optimizer updates, no trained neighborhood prior.
- Tests: `tests/test_i10_track_neighborhood.py` (11), `test_i10_track_visible.py` (7).
  Original PIDG and all frozen historical modules/weights remain unchanged.
'''
    patches=[]
    for name,addition in [('experiments/adaptive_reconstruction_plan.md',section),('experiments/staged_reconstruction_plan.md',section),('PIDG_REPO_MAP.md',mapping)]:
        path=ROOT/name;head=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8')
        current=path.read_text(encoding='utf-8');assert addition.splitlines()[0] not in current
        path.write_text(current.rstrip()+'\n\n'+addition,encoding='utf-8',newline='\n')
        updated=head.rstrip()+'\n\n'+addition
        patches.extend(difflib.unified_diff(head.splitlines(True),updated.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    dest=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_track_neighborhood/results_cached.patch'
    dest.write_text(''.join(patches),encoding='utf-8',newline='\n')
    subprocess.run(['git','apply','--cached','--whitespace=nowarn',str(dest)],cwd=ROOT,check=True)


if __name__=='__main__':main()
