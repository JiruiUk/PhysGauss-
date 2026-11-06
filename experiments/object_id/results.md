# C1 self-supervised Object ID comparison

Status: completed | Active: O2_selfsup_ID / metrics_test

MovingParts-inspired shared object motion + retained nonrigid C1 path. No instance labels or pseudo-ID targets.
C1 alpha/RGB/perceptual recipe retained in all arms; not a fully RGB-only experiment.
Matched full3k prefix; total50k each;120 train /30 validation /50 final test.
O0 reuses completed C1 weights, validation history, metrics and renders; only O1/O2 train.
16 candidate slots are a capacity ceiling, not a known or forced object count. Effective occupancy is a diagnostic.

| Model | Current iter | Validation PSNR | Final test PSNR | Effective slots >1% |
|---|---:|---:|---:|---:|
| O0_C1_control (reused) | 50000/50000 | 39.664894612630206 | 40.372127532958984 | - |
| O1_frozen_ID | 50000/50000 | 38.27950229644775 | 38.81082534790039 | 16.0 |
| O2_selfsup_ID | 50000/50000 | 38.0785613377889 | 38.480873107910156 | 16.0 |

[Monitor](monitor.html) | [Plan](plan.json) | [Preflight](preflight.json)

GT/render export after training: F:\A_Edinburgh\PhysGauss\runs\pidg_object_id\render_exports\final50000\gt_c1_frozen_learned
