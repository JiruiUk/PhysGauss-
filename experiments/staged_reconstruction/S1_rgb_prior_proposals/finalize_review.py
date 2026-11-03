"""Archive D4 negative result and task-only shared plan/map sections."""
import difflib
from pathlib import Path
import subprocess
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.rgb_prior_proposals.prepare import REPORT, RUNS, verify

SECTION = '''

### S1-D4：备选充分但RGB代价几乎全拒，未见收益（2026-11-03）

按59fb74b3单候选协议实现/冻结aabe584b并完成两例各96图。22测试、独立进程
2+2保存恢复、全部192图输入/界/接受方程与实际view82原RGB代价重放均通过。
全部423747/423100候选保留，有限备选38290/37475，但fit不恶化仅3/2、最终
接受1/1。唯一c26_t3(85,108)GT误差0.009416→0.039775，held代价下降亦不能
证明几何正确。全局MAE约0.036805/0.036826不改善，薄结构准确覆盖约41.8%/
41.6%不变；数值及Gaussian证据门槛均失败。固定图及唯一变化裁切实际已看。
不放宽阈值/容差、不更换模型，不做当前Gaussian或旧真实续训，无新86/188。
下一步仅评估专用RGB代价oracle审计：对比连续GT/理想同格GT/D1代价，并按
跨面窗口、源真实可见性分组，区分匹配测量/遮挡/搜索瓶颈。GT不得输出为
训练深度/法线/掩膜/系数/选点；未做该审计前不预设是某一个原因或宣称RGB无解。
见[D4评估](staged_reconstruction/S1_rgb_prior_proposals/v1/assessment.md)。
'''
MAP = '''

## Optional RGB-validated relative-depth proposal (2026-11-03)

`variants/staged_reconstruction/rgb_prior_proposals/core.py` uses frozen raw
Small ordering and D2 interval anchors to nominate alternative192-grid depths;
the unchanged patch_features/source_cost and fixed4-fit/2-held checks accept
or retain old D1 depths. No learned static label or physical identity is made.
`compute/infer/io` retain full membership with atomic hash-bound per-view saves;
`preflight` verifies independent2+2 inference recovery and exact disabled identity.
`audit` replays every saved input/bound/acceptance and actual native RGB costs in
a separate process; `evaluate` alone reads static GT and visible-thin labels.
No PIDG renderer/model/optimizer/physics or Gaussian training integration.
Tests: test_rgb_prior_proposals.py; reports: S1_rgb_prior_proposals/v1.
'''


def main():
    verify(); gate = read(REPORT/'gate.json'); assert not gate['both_cases_numeric_improvement'] and not gate['both_cases_Gaussian_evidence']
    assert not (REPORT/'final_review.json').exists()
    figures = {**read(REPORT/'evaluation.json')['images'], **read(REPORT/'changed_pixel_review.json')['images']}
    for path, h in figures.items(): assert sha(ROOT/path) == h, path
    review = {'status': 'complete_rejected_no_numeric_improvement', 'manual_review_completed': True,
        'actually_viewed': 'Both cases c00_t1/c08_t0/c24_t0 RGB/D1/D4/error-change images and sole c26_t3 changed-pixel full RGB/x4 crop',
        'finding': 'Many finite proposals, native fit cost almost always vetoes; sole accepted point worsens independent GT error.',
        'original_membership_unaccepted_depths_and_thresholds_preserved': True,
        'Gaussian_training_started': False, 'real_training_authorized': False,
        'next': 'Preregister evaluator-only native photometric-cost oracle/visibility/cross-face audit; no GT assets exported or gate relaxation',
        'reviewed_images': figures, 'evidence_sha256': {name: sha(REPORT/name) for name in
            ['source_freeze.json','preflight.json','inference.json','input_audit.json','rejection_accounting.json','evaluation.json','gate.json']}}
    write(REPORT/'final_review.json', review)
    write(RUNS/'final_status.json', {'status': review['status'], 'review_sha256': sha(REPORT/'final_review.json'),
        'Gaussian_training_started': False, 'next': review['next']})
    patches = []
    for name, section in [('experiments/adaptive_reconstruction_plan.md', SECTION), ('experiments/staged_reconstruction_plan.md', SECTION), ('PIDG_REPO_MAP.md', MAP)]:
        original = subprocess.check_output(['git','show','HEAD:'+name], cwd=ROOT).decode('utf-8').replace('\r\n','\n')
        path = ROOT/name; working = path.read_text(encoding='utf-8').replace('\r\n','\n')
        assert section.strip() not in working
        path.write_text(working+section, encoding='utf-8', newline='\n')
        patches.extend(difflib.unified_diff(original.splitlines(True), (original+section).splitlines(True), fromfile='a/'+name, tofile='b/'+name))
    (ROOT/'tmp/S1_rgb_prior_proposals_plans.patch').write_text(''.join(patches), encoding='utf-8', newline='\n')
    print('RGB_PRIOR_PROPOSAL_REVIEW_COMPLETE', flush=True)


if __name__ == '__main__': main()
