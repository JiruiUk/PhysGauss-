"""Archive completed D3 and produce task-only shared-file cached patch."""
import difflib
import subprocess
from pathlib import Path
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.pretrained_depth.prepare import REPORT, RUNS, verify

SECTION = '''

### S1-D3：固定Small已测，局部锚点无法标定全图（2026-11-03）

用户选定Depth Anything V2 Small后，按e47030ac预声明协议实际完成两例各96图。
官方代码/权重SHA冻结，27测试、192图输入/锚点/标定重放及独立进程模型重载
逐值通过。仅1/96图/例通过空间检查锚点标定；其可见静态深度MAE0.116/0.112，
且该图100%真空背景被赋有限深度。两例直接静态监督门槛均失败，无Gaussian/
真实训练、新86/188/PLY/视频，不放宽旧门槛或更换候选。实际固定0/24/72与
唯一成功c26_t2的RGB/原始先验/误差图均已查看，不能归因反光或宣称数据无解。
原始深度顺序与GT相关均值约0.896，但不是几何正确/静态身份的证明。
下一步固定顺序可靠性及评估专用理想尺度诊断，区分标定瓶颈与单目形状偏差；
GT仅诊断，不回灌系数/掩膜/训练点。不能以排序均值绕过Gaussian完整性门槛。
见[Small实测](staged_reconstruction/S1_pretrained_geometry/v1/assessment.md)。
'''
MAP = '''

## Optional fixed pretrained RGB observed-depth evidence (2026-11-03)

`variants/staged_reconstruction/pretrained_depth/model.py` lazily loads the
hash-verified official Small model with weights_only and unchanged518 BGR
preprocessing; disabled calls retain object identity without model import.
`core.py` uses disjoint spatial train-anchor checks and Theil-Sen inverse-depth
calibration, retaining failed views as unknown. `infer/prepare/replay` freeze
RGB/anchors/upstream source and replay official weights in a separate process.
`evaluate` alone imports analytic GT ray tracing, checks static/thin/observed
depth, true empty background and same-pose time overlap. No original PIDG
renderer, Gaussian, optimizer, physics field or identity is changed. Tests:
test_pretrained_depth.py. Reports: S1_pretrained_geometry/v1; not Gaussian training.
'''


def main():
    verify(); gate = read(REPORT/'gate.json'); evaluation = read(REPORT/'evaluation.json')
    assert not gate['both_cases_evidence_passed'] and not (REPORT/'final_review.json').exists()
    images = {**evaluation['images'], **read(REPORT/'supplemental_visuals.json')['images']}
    for p, h in images.items(): assert sha(ROOT/p) == h, p
    review = {'status': 'complete_rejected_direct_static_depth', 'manual_review_completed': True,
        'actually_viewed': 'Both cases c00_t1/c08_t0/c24_t0 plus sole accepted c26_t2 RGB/raw/metric/GT-error panels',
        'finding': 'Strong coarse ordering, local anchor calibration fails most views; accepted view extrapolation and empty background unsafe.',
        'no_Gaussian_or_real_training': True, 'no_threshold_weight_or_candidate_changes': True,
        'next': 'Preregister relative-order reliability and evaluator-only oracle metric-shape diagnostic; no GT feedback to training',
        'evidence_sha256': {n: sha(REPORT/n) for n in ['inference.json','evaluation.json','gate.json','source_freeze.json','input_audit.json','replay.json']},
        'reviewed_images': images}
    write(REPORT/'final_review.json', review)
    write(RUNS/'final_status.json', {'status': review['status'], 'review_sha256': sha(REPORT/'final_review.json'),
        'Gaussian_training_started': False, 'next': review['next']})
    patches = []
    for name, section in [('experiments/adaptive_reconstruction_plan.md', SECTION), ('experiments/staged_reconstruction_plan.md', SECTION), ('PIDG_REPO_MAP.md', MAP)]:
        original = subprocess.check_output(['git','show','HEAD:'+name], cwd=ROOT).decode('utf-8').replace('\r\n','\n')
        path = ROOT/name; working = path.read_text(encoding='utf-8').replace('\r\n','\n')
        assert section.strip() not in working
        path.write_text(working+section, encoding='utf-8', newline='\n')
        patches.extend(difflib.unified_diff(original.splitlines(True),(original+section).splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    (ROOT/'tmp/S1_pretrained_geometry_plans.patch').write_text(''.join(patches),encoding='utf-8',newline='\n')
    print('PRETRAINED_DEPTH_REVIEW_ARCHIVED', flush=True)


if __name__ == '__main__': main()
