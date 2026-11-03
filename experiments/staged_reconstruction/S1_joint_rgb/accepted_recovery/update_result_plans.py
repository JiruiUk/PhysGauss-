"""Record completed J1 evidence as task-only shared-plan patches."""
import difflib
import subprocess
from pathlib import Path
from variants.staged_reconstruction.prepare_data import ROOT

SECTION = '''### S1-J1-A完成：RGB清理改善，但准确深度覆盖退化，不升级（2026-11-03）

四臂各6000完整Gaussian训练、两Adam/RNG/6000相机采样、独立GT评估和原生
导出661d3158完成。原控制静态精确，候选仅opacity/SH变更，22个动态Adam
状态到6000；8PLY与检查点匹配、两H264各解码160帧。实际看两例静态对比、
训练0/24/72、novel24五时间、原生t2、D/alpha/alpha及解码图，pending已消解。

静态GT PSNR34.408→36.819、33.877→36.065dB，RGB外溢减少70.28/68.87%；
但以47782个GT静态像素为分母的准确射线覆盖88.996→84.618%、87.376→82.378%，
损失4.378/4.998百分点，原<=1门槛失败。薄结构中心覆盖大体保留不抵消它。
alpha缺失仅19→112、33→114，不能把全部覆盖回退称为大面积破洞。RGB悬浮块
变暗，而alpha/深度仍有悬浮支撑和底座外晕；动态池含静态柱/底座，部分动态
球仍是亮斑。不能从外观改善认定几何已清理，不准入Rubber Duck或旧真实续训。

下一步固定一个无训练分解审计：完整lost/gained射线闭合分母，区分alpha缺失
与可见深度偏移；原/candidate opacity×原/candidate SH固定2×2反事实，仅作
冻结权重诊断，不冒充四种新训练或完整训练因果。GT仅在独立评估器中，不输出
训练选点/掩膜/监督，不放宽门槛或扫损失权重。依据证据再定一个结构保护候选。
见S1_joint_rgb/accepted_recovery/v1/assessment.md；完整失败和用户恢复授权保留。'''


def main():
    chunks = []
    for name in ['experiments/adaptive_reconstruction_plan.md', 'experiments/staged_reconstruction_plan.md']:
        head = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT).decode('utf-8')
        assert SECTION.splitlines()[0] not in head
        final = head.rstrip()+'\n\n\n'+SECTION+'\n'
        path = ROOT/name; current = path.read_text(encoding='utf-8')
        assert SECTION.splitlines()[0] not in current
        path.write_text(current.rstrip()+'\n\n\n'+SECTION+'\n', encoding='utf-8', newline='\n')
        chunks.extend(difflib.unified_diff(head.splitlines(keepends=True), final.splitlines(keepends=True),
            fromfile='a/'+name, tofile='b/'+name))
    patch = ROOT/'runs/pidg_staged_reconstruction/S1_joint_rgb/accepted_recovery/result_plans_cached.patch'
    patch.write_text(''.join(chunks), encoding='utf-8', newline='\n'); print(patch)


if __name__ == '__main__': main()
