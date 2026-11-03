"""Persist D2 manual review and task-only shared-file updates, without gate edits."""
import difflib
from pathlib import Path
import subprocess
from variants.staged_reconstruction.depth_uncertainty.prepare import BASE,REPORT,RUNS,verify
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha

SECTION = '''

### S1-D2：源视角不确定性有辨别力，但覆盖门槛未过（2026-11-03）

按3b69cf52预声明协议完成两例各96视图，21测试与冻结/父深度/flags-off审计
通过；全部423747/423100候选及深度保留。经验源包络不是概率置信区间。
严格支撑GT区间覆盖95.86%/95.96%，大错误0.69%/0.49%，但候选保留仅
2.03%/2.01%、薄结构支撑保留3.17%/3.15%，未达到预定25%覆盖门槛。
不放宽区间/机位/门槛，不启动当前区间损失Gaussian训练或旧真实续训。
实际c00_t1/c08_t0/c24_t0诊断图已看：可靠点集中在立柱少量面片，底座/
边界支撑仍不足。AUROC约0.76不是静态身份或正确几何证明，亦非伪影改善。

当前已测RGB-MVS证据链存在可靠性/覆盖瓶颈，不证明所有RGB方法都不可解。
用户已明确选择固定Depth Anything V2 Small预训练RGB几何先验；据此继续
新代码/权重冻结和两例独立GT审计，无静态标签/物体数，不再次等待确认。
先验仍非GT，不升级S3-U；新Gaussian训练须另立固定协议和原生完整恢复/
静态GT完整性门槛。见[D2评估](staged_reconstruction/S1_depth_uncertainty/v1/assessment.md)
及[已选下一方向](staged_reconstruction/S1_depth_uncertainty/next_choice.md)。
'''
MAP = '''

## Optional RGB source-depth interval evidence (2026-11-03)

`variants/staged_reconstruction/depth_uncertainty/core.py` observes the original
six RGB-derived source depths, counts distinct poses and constructs empirical
depth envelopes with an inverse-depth quantization floor. Missing/closer-surface
samples retain unknown status. Disabled calls return the original depth object.
`infer` preserves all D1 candidates/depths with per-view stop/lock/source checks.
`evaluate` alone reads GT and measures reliability, width and visible/thin
retention; `prepare` binds tests, protocol, parent arrays and frozen source.
This is a geometry-evidence diagnostic, with no change to PIDG models, fields,
renderer, optimizer or physical identities. Tests: test_depth_uncertainty.py.
'''


def main():
    verify(); gate = read(REPORT/'gate.json'); evaluation = read(REPORT/'evaluation.json')
    assert not gate['both_cases_evidence_passed'] and not (REPORT/'final_review.json').exists()
    for p,h in evaluation['images'].items(): assert sha(ROOT/p) == h,p
    review = {'status':'complete_rejected_coverage','manual_review_completed':True,'Gaussian_training_started':False,
        'real_training_authorized':False,'thresholds_changed':False,'all_candidates_and_depth_preserved':True,
        'actually_viewed':'Diffuse/glossy c00_t1/c08_t0/c24_t0 native-size center-error/evidence-score/interval-width/tight-GT panels',
        'finding':'Tight evidence is accurate but concentrated on a few column patches; base/boundary and thin retention insufficient.',
        'not_a_data_impossibility_or_Gaussian_cleanup_claim':True,
        'user_choice_received':'Fixed Depth Anything V2 Small RGB geometry prior; no static labels or object count',
        'next':'New fixed prior protocol, official code/weights freeze, data-only input boundary and independent synthetic GT audit; no repeated confirmation.',
        'evidence_sha256':{n:sha(REPORT/n) for n in ['inference.json','evaluation.json','gate.json','source_freeze.json','test_receipt.json']},
        'reviewed_images':evaluation['images']}
    write(REPORT/'final_review.json',review)
    write(RUNS/'final_status.json',{'status':review['status'],'review_sha256':sha(REPORT/'final_review.json'),
        'next_authorized':'Fixed DA-V2 Small prior audit','Gaussian_training_started':False})
    patches = []
    for name,section in [('experiments/adaptive_reconstruction_plan.md',SECTION),('experiments/staged_reconstruction_plan.md',SECTION),('PIDG_REPO_MAP.md',MAP)]:
        original = subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8').replace('\r\n','\n')
        path = ROOT/name; working = path.read_text(encoding='utf-8').replace('\r\n','\n')
        assert section.strip() not in working
        path.write_text(working+section,encoding='utf-8',newline='\n')
        patches.extend(difflib.unified_diff(original.splitlines(True),(original+section).splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    (ROOT/'tmp/S1_depth_uncertainty_plans.patch').write_text(''.join(patches),encoding='utf-8',newline='\n')
    print('D2_REVIEW_COMPLETE_NEXT_PRIOR_AUTHORIZED',flush=True)


if __name__ == '__main__': main()
