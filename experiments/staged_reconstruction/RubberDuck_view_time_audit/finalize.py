"""Finalize evidence already measured and visually inspected; no GPU or training."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
REPORT=BASE/'v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_view_time_audit/v1'
RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_view_time_audit/v1'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

assert json.loads((REPORT/'final_validation.json').read_text())['passed']
assert json.loads((REPORT/'independent_audit.json').read_text())['passed']
viewed=[]
for model in ['J7','rgb_control','temporal_memory']:
    for a in [24,60,95]:
        for kind in ['joint_cross','alpha_cross','observed']:
            viewed.append(OUT/'figures'/f'{model}_a{a}_{kind}.png')
for a in [24,60,95]:viewed.append(OUT/'figures'/f'a{a}_gray_comparison.png')
for a in [60,95]:viewed.append(OUT/'figures'/f'a{a}_joint_comparison.png')
for kind in ['angle_sweep','paired_controls']:viewed.append(OUT/'figures'/f'{kind}.png')
for a in [24,60,95]:
    for kind in ['preview','explanation']:viewed.append(OUT/'figures'/f'J7_a{a}_{kind}.png')
for name in ['J7/native1600/c024_t024_joint.png','J7/native1600/c024_t024_gray.png',
             'J7/native1600/c060_t060_gray.png','J7/native1600/c084_t060_dynamic.png',
             'temporal_memory/native1600/c060_t060_gray.png','temporal_memory/native1600/c071_t095_dynamic.png']:
    viewed.append(OUT/name)
assert len(viewed)==46
write(REPORT/'manual_review.json',dict(completed=True,reviewed_files={p.relative_to(ROOT).as_posix():sha(p) for p in viewed},
    reviewed_images=46,new_native_measurement_after_selection=False,
    findings=[
        'RGB off-diagonal fragmentation is clear particularly at middle/late anchors; early combinations are not uniformly worse.',
        'Observed RGB is relatively complete but gray dynamic support already includes elongated/large dark support at matched cameras.',
        'Static-free constant-gray dynamic rendering retains deformed support artifacts; appearance/static compositing alone is insufficient.',
        'Both J8 continuations retain the main view-time-local pattern; no structural promotion.',
        'Dark/high-alpha fractions are not monotonic quality scores: they can decrease while visible RGB fragments.',
        'No simultaneous novel-view GT; perspective, self-occlusion and time-varying motion remain confounds.'
    ],same_time_novel_view_GT=False,new_training_updates=0))
write(REPORT/'figure_receipts.json',dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted((OUT/'figures').glob('*.png'))}))
(BASE/'progress.md').write_text('''# V0 complete: read-only camera/time diagnostic

2026-11-06. Protocol 91248fce; implementation/qualification 76ddadcf;
full measurements and independent replay 354056b2.

- Ten tests and all three native/RNG/flags/exception/full-state checks passed.
- 544 input/source files frozen before measurement; exact J7 plus both J8 arms.
- 482 pairs/model, 1446 unique model/pairs, 63 native1600 combinations.
- Independent second-process replay of every800 metric and every800/1600 raw
  render hash passed; 4710 output PNG receipts and full denominator validation.
- Actual visual review: nine RGB crossing sheets, nine alpha crossing sheets,
  nine observation controls, three gray comparisons, two RGB comparisons,
  two numerical curves, six presentation/preview sheets and six native images.
- The view-time-local RGB advantage is supported, especially middle/late, but
  matched-camera gray/alpha also contain large dark support. Not appearance-only.
- Dark fractions are not monotonic structure scores. No simultaneous novel-view
  GT, no formal true generalization error, no unique cause established.
- J8 memory has not removed the pattern. Retain J7; zero training/model changes.

Assessment, manual review and final_review.json close pending review. This
diagnostic is complete; no automatic training, candidate sweep or paused resume.
''',encoding='utf-8')
# Only our own append is staged via the generated cached-only patch. Preserve
# every pre-existing working-tree edit, including unrelated changes in these docs.
updates={
    'experiments/adaptive_reconstruction_plan.md': '''
## V0 camera/time coupling check completed (2026-11-06)

Preregistered read-only J7/J8 audit: 482 pairs/model, 1446 model/pairs and63
native1600 combinations; all second-process native hashes/metrics and full-state
no-mutation checks pass. RGB completeness is localized near observed camera/time
combinations, especially middle/late; gray dynamic support retains large dark
elongated support even at matched views. Both J8 continuations retain this pattern.
This supports trajectory-specific fitting/underconstrained geometry, without
simultaneous novel-view GT or proof of a sole cause. Dark fractions are nonmonotonic
and cannot rank structural accuracy. No new training/promotion; retain J7.
See staged_reconstruction/RubberDuck_view_time_audit/v1/assessment.md.
''',
    'experiments/staged_reconstruction_plan.md': '''
## V0：J7/J8相机—时间交叉检查（2026-11-06，完成）

三个检查点各482组、共1446组交叉渲染及63组原生1600导出；独立进程
全部RGB/alpha/D哈希、分母与完整状态复核通过。120图=120时间=120相机，
同时间多相机0。中晚期可见外观在观测组合附近较完整，偏离组合后碎裂；
原视角灰色动态支撑也存在大片拖尾，不能只归因SH或静态遮挡。J8两臂仍有
此现象；暗高alpha比例下降不代表结构变好。支持轨迹相关拟合/几何欠约束，
无同时间新视角GT，不给真实准确率或唯一因果判断。保留J7，0训练/0晋级。
完整报告：staged_reconstruction/RubberDuck_view_time_audit/v1/assessment.md。
''',
    'PIDG_REPO_MAP.md': '''
## Read-only camera/time coupling audit V0 (2026-11-06)

`variants/staged_reconstruction/view_time_audit/core.py` owns pair deduplication,
actual-pose distances, undefined-aware dark/support proxies and angular accounting.
`worker.py` defaults off and loads exact J7/J8 full states via the frozen adapter;
native joint/dynamic/gray renders preserve both Adam/RNG/camera/physics/controller
states, including exception-path flags. Independent process replays every metric
and raw render hash. Reports/presentation/independent accounting live in
experiments/staged_reconstruction/RubberDuck_view_time_audit. No new training,
physics, labels, instances or model parameters; proxy support is not geometry GT.
'''}
patches=[]
for rel,addition in updates.items():
    baseline=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT).decode('utf-8')
    patches.extend(difflib.unified_diff(baseline.splitlines(True),(baseline+addition).splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
    path=ROOT/rel;old=path.read_bytes();marker=addition.strip().splitlines()[0].encode('utf-8')
    assert marker not in old,rel
    newline=b'\r\n' if b'\r\n' in old else b'\n'
    path.write_bytes(old+addition.encode('utf-8').replace(b'\n',newline))
(BASE/'shared_cached.patch').write_text(''.join(patches),encoding='utf-8',newline='\n')
artifacts={p.name:sha(p) for p in REPORT.iterdir() if p.is_file() and p.name!='final_review.json'}
write(REPORT/'final_review.json',dict(status='completed_read_only_diagnostic',pending=False,pending_review=False,
    training_updates=0,models_promoted=False,J7_retained=True,all_native_replay_passed=True,
    full_state_unchanged=True,observation_supported=True,unique_overfitting_cause_proven=False,
    novel_view_independent_GT=False,reviewed_images=46,artifacts=artifacts))
final=dict(status='completed_read_only_diagnostic',pending=False,pending_review=False,training_updates=0,
    models_promoted=False,source_sha256=sha(REPORT/'source_freeze.json'),review_sha256=sha(REPORT/'final_review.json'))
for name in ['final_status.json','active.json','state.json','pipeline.json']:write(RUNS/name,final)
print('V0_FINAL_REVIEW_COMPLETE 46 viewed, zero updates')
