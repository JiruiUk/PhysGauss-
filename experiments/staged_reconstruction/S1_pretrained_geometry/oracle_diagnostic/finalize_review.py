"""Archive actual diagnosis/image review; stage only this task's shared sections."""
from pathlib import Path
import subprocess
import difflib
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.pretrained_depth.prepare import RUNS
from experiments.staged_reconstruction.S1_pretrained_geometry.oracle_diagnostic.run import DIAG, check

SECTION = '''

### D3-O：单目形状/薄结构顺序也有偏差，转向RGB验证的候选（2026-11-03）

固定Small的192个输出已做预声明的独立形状/顺序诊断（3fac945b/5c311055，20测试）。
仅评估可用的GT仿射对齐后，留出可见静态MAE仍约0.0722；相同D1像素约0.0677，
D1约0.0357。特权对齐不是训练或部署结果，系数/深度/掩膜未导出。强顺序的
静态错误约2.2%、薄结构约8.7%；强候选仅7.42%两端均可见静态，91.24%涉及
真空背景（运动/背景类别可重叠）。不能把整体排序相关0.896当静态/细结构监督。
两例固定RGB/误差及明确交集补图已实际查看，原图/指标/门槛保留。
继续一个单独预声明的S1_rgb_prior_proposals候选：先验只提出RGB深度备选，原
候选全部保留，由固定fit/held源RGB验证；不直接增加metric/法线/全图排序损失。
不更换模型/扫阈值，不进入旧真实续训；Gaussian阶段仍需独立完整性门槛。
见[诊断评估](staged_reconstruction/S1_pretrained_geometry/oracle_diagnostic/assessment.md)。
'''


def main():
    check(); assert not (DIAG/'final_review.json').exists()
    evaluation = read(DIAG/'evaluation.json'); figures = read(DIAG/'figure_clarification.json')
    images = {**evaluation['images'], **figures['images']}
    for p, h in images.items(): assert sha(ROOT/p) == h, p
    assert not list(DIAG.glob('*.npz'))
    review = {'status': 'complete_diagnostic_no_training_admission', 'manual_review_completed': True,
        'actually_viewed': 'Both cases fixed c00_t1/c08_t0/c24_t0 input/all-held-error and explicit common-D1-pixel error panels',
        'finding': 'Affine anchor scarcity is not sole issue; remaining shape bias and thin order errors forbid direct static-depth or full-image ordinal loss.',
        'GT_fit_coefficients_arrays_masks_not_exported': True, 'original_D3_predictions_and_gate_preserved': True,
        'next': 'One separately preregistered RGB-validated ordinal proposal, retaining all D1 candidates; no direct prior loss',
        'reviewed_images': images, 'evaluation_sha256': sha(DIAG/'evaluation.json'),
        'assessment_sha256': sha(DIAG/'assessment.md')}
    write(DIAG/'final_review.json', review)
    write(RUNS/'oracle_diagnostic_final_status.json', {'status': review['status'], 'review_sha256': sha(DIAG/'final_review.json'),
        'Gaussian_training_started': False, 'next': review['next']})
    patches = []
    for name in ['experiments/adaptive_reconstruction_plan.md', 'experiments/staged_reconstruction_plan.md']:
        original = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT).decode('utf-8').replace('\r\n','\n')
        path = ROOT/name; working = path.read_text(encoding='utf-8').replace('\r\n','\n')
        assert SECTION.strip() not in working
        path.write_text(working+SECTION, encoding='utf-8', newline='\n')
        patches.extend(difflib.unified_diff(original.splitlines(True), (original+SECTION).splitlines(True), fromfile='a/'+name, tofile='b/'+name))
    (ROOT/'tmp/S1_pretrained_diagnostic_plans.patch').write_text(''.join(patches), encoding='utf-8', newline='\n')
    print('D3_ORACLE_DIAGNOSTIC_REVIEW_COMPLETE', flush=True)


if __name__ == '__main__': main()
