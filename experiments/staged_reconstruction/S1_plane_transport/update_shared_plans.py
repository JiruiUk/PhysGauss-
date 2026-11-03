"""Append only this task's evidence; construct cached-only changes from HEAD."""
import difflib
from pathlib import Path
import subprocess
from variants.staged_reconstruction.prepare_data import ROOT

PLAN_SECTION = '''### S1-D7：跨视角传递有局部收益，未准入；转查联合撤销路径（2026-11-03）

7680e19a固定协议完整执行，实现/40测试/809文件冻结及三独立进程完整4对2+2
恢复9b2d4ed4；两条件各96图推断、全192父值/源查询/平面/接受/hash、六固定
原RGB重放与独立GT/分母/MAE/P90/原始数据SHA核验eca285b4。全部423747/423100
候选保留，58279/58201更新，其中25119/25226没有自身合格斜率也获外视角更新。
静态MAE0.03659113→0.03590450、0.03661146→0.03591997（约1.88/1.89%）；大错误
净减少603/602，仍占20.13/20.15%。准确静态覆盖60.37/60.27%，薄结构42.66/42.47%，
薄结构MAE/P90均略改善，但原数值5%和Gaussian严格证据门槛均失败，不升级。
约94.9%的父大错误仍无可用源平面；不放宽原邻域、支撑范围或RGB容差，不重跑。
实际查看两例固定0/24/72原图/误差/更新及全部解析控制。final_review/status已
消解pending；无Gaussian/Rubber Duck新训练、原生86/188、PLY或视频结果。

用户提出动态阶段联合清理静态拖影。现S3-U全冻结静态，原S6只放开静态颜色；
动态拟合不能直接撤销静态opacity。并行static_dynamic_coupling审计9a9c8fb6已
检查初始化/梯度/遮挡，去掉静态后动态仍碎裂，不把所有缺失归于当前静态遮挡，
不重复该审计。下一单阶段J0仅检查全RGB无颜色掩膜下开放静态opacity/SH的
原生梯度与状态可恢复性，无优化器更新、不拼接新旧静态库、保留旧权重条件
颜色训练的来源说明。它不是联合训练成功或未知静态身份推断，不能绕过失败
深度/Gaussian门槛。今后学习候选仍须另立匹配协议、完整Adam/RNG/相机恢复及
两条件独立静态GT/结构审计后再考虑Rubber Duck。S3-U/Small/未知身份与暂停
任务保留，不新增本构/材料/力/接触、不扫参。
见[D7评估](staged_reconstruction/S1_plane_transport/v1/assessment.md)、
[后续方向](staged_reconstruction/S1_plane_transport/next_plan.md)和
[J0固定协议](staged_reconstruction/S1_joint_revocation_audit/protocol.json)。'''

MAP_SECTION = '''## Opt-in cross-view source-plane transport (S1-D7, 2026-11-03)

`variants/staged_reconstruction/plane_transport/core.py` queries original first4
fit-source anchors, transforms original D1/D6-admitted planes and intersects
target rays inside source5x5 footprints. Target slope admission is unnecessary;
source checks and frozen raw-RGB warp/fit/held equations are unchanged. Disabled
calls return the same parent object. inputs/compute retain all queries, source
planes, rejected hypotheses and parent rows. infer is locked/default-off;
preflight checks independent full4 versus2+2 for both conditions. prepare binds
809 train-only/source files. audit replays192 states plus six full RGB views;
evaluate alone reads static/visible-thin GT. Independent export_review replays
aggregate counts/MAE/P90 and original GT hashes; no GT feedback. 40 tests pass.
D7 has local depth gains but fails both gates, not integrated into Gaussian
training. No PIDG trainer/renderer, frozen source, weight or physics changes.
Reports: S1_plane_transport/v1. Next J0 is preregistered no-update full-RGB
static-opacity/SH gradient audit; no joint-training implementation claimed.'''


def transform(text, section, is_map=False):
    if is_map:
        text = text.replace('S1-D7 source-plane transport is preregistered only, not implemented here yet.',
            'S1-D7 is implemented and completed; see the following map section.')
    assert section.splitlines()[0] not in text, 'Do not append duplicate task evidence'
    return text.rstrip()+'\n\n\n'+section.strip()+'\n'


def main():
    chunks = []
    for name, section in [('experiments/adaptive_reconstruction_plan.md', PLAN_SECTION),
        ('experiments/staged_reconstruction_plan.md', PLAN_SECTION), ('PIDG_REPO_MAP.md', MAP_SECTION)]:
        head = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT).decode('utf-8')
        final_head = transform(head, section, name == 'PIDG_REPO_MAP.md')
        path = ROOT/name
        current = path.read_text(encoding='utf-8')
        if section.splitlines()[0] in current:
            assert current.rstrip().endswith(section.strip()), 'Unexpected partial task section'
        else:
            path.write_text(transform(current, section, name == 'PIDG_REPO_MAP.md'), encoding='utf-8', newline='\n')
        chunks.extend(difflib.unified_diff(head.splitlines(keepends=True), final_head.splitlines(keepends=True),
            fromfile='a/'+name, tofile='b/'+name))
    patch = ROOT/'runs/pidg_staged_reconstruction/S1_plane_transport/v1/shared_plans_cached.patch'
    patch.write_text(''.join(chunks), encoding='utf-8', newline='\n')
    print(patch, flush=True)


if __name__ == '__main__':
    main()
