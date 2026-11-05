"""Task-only final plan additions; unrelated working edits stay unstaged."""
import difflib
from pathlib import Path
import subprocess
from variants.staged_reconstruction.i10_tracking.io import ROOT,read


def main():
    b=ROOT/'experiments/staged_reconstruction/RubberDuck_I10_tracking/training/v1'
    final=read(b/'final_review.json');assert not final['pending_review']
    ev=read(b/'evaluation.json');metrics=ev['validation']['means']
    fit=ev['training_track_diagnostic']['models'];manual=read(b/'manual_review.json')
    section=f'''### I10无颜色阈值三维轨迹续训完成（T0–T2，2026-11-05）

用户要求去掉颜色阈值静态参考，尝试移动物体3D追踪。新监督仅来自全图RGB、
已知相机/时间和固定CoTracker3对应；无新颜色/语义/alpha前景掩膜、物体数或GT。
T0稀疏提点32绑定（23/5/4）未过覆盖，结果保留。单个密集采样修正T1保持全部
可靠性/歧义/训练准入门槛：4433提案、730一致轨迹、222运动候选、123绑定到
116个持续高斯ID（98/11/14）；全部28窗口原始轨迹和120帧父三维位置/绑定
独立进程逐值重放通过。3D深度仍依赖模型，不是移动单目静态三角化GT。

T2从J6最终联合模型完整双Adam/RNG/相机/physics缓存起步，静态与physics全冻，
两臂各2000原生1600更新；RGB原图0.8L1+0.2DSSIM，候选另加固定0.01稳健
持续三维点重投影。1432去重观测覆盖96/120训练帧；无旧I10静态目标替换/
前景/coverage，也无静态自深度损失。父仍有颜色条件训练历史，不能称从零
未知静态发现。19测试、原生flags-off/3D梯度及四组跨进程完整精确状态与
固定native梯度Adam恢复通过；保留原CUDA轨迹敏感性，单对照不证明稳定因果。

30张val平均PSNR：J6父{metrics['J6_parent']['PSNR']:.6f}，RGB对照
{metrics['rgb_only']['PSNR']:.6f}，RGB+轨迹{metrics['rgb_tracks']['PSNR']:.6f}dB，差
{metrics['rgb_tracks']['PSNR']-metrics['rgb_only']['PSNR']:+.6f}dB。
训练对应平均误差{fit['J6_parent']['mean_px']:.6f}→{fit['rgb_only']['mean_px']:.6f}→
{fit['rgb_tracks']['mean_px']:.6f}px只是拟合诊断。静态模型/Adam/RGB/alpha/D与
physics状态精确不变。原生86/188早中晚、六训练观测、完整Gaussian/世界XYZ
PLY和两200帧H264均复核，实际人工结论：{manual['conclusion']}。
状态{final['status']}；未放宽原独立几何/完整性门槛。细节见
RubberDuck_I10_tracking/training/v1/assessment.md。结束本轮，不自动延长或调权，
不恢复旧PAUSED任务；未观测/遮挡/歧义保留未知，稀疏追踪不是完整结构证明。
'''
    patches=[]
    for name in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']:
        path=ROOT/name;head=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8')
        current=path.read_text(encoding='utf-8');assert section.splitlines()[0] not in current
        updated=head.rstrip()+'\n\n'+section
        path.write_text(current.rstrip()+'\n\n'+section,encoding='utf-8',newline='\n')
        patches+=list(difflib.unified_diff(head.splitlines(True),updated.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    dest=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_tracking/final_plans_cached.patch';dest.write_text(''.join(patches),encoding='utf-8',newline='\n')
    subprocess.run(['git','apply','--cached','--whitespace=nowarn',str(dest)],cwd=ROOT,check=True)

if __name__=='__main__':main()
