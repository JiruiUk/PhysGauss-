# I2 sparse shared motion — rejected at cross-view screen

The formal run completed 2,000 actual updates from the same immutable S3-U
final6k state as I0/I1. Native rendering and physics query the same blended
center mapping. There were 1,601 fully sparse updates and 200 inherited physics
calls. The full-state, camera-sequence, frozen-static and source checks passed.
Checkpoint SHA256: `c43d589a7c6918ff96e01511d9441a2d919921f06f97db7353e444dc4ca59863`.

| Native1600 validation (30 frames) | S3-U | I0 | I1 | I2 |
|---|---:|---:|---:|---:|
| PSNR | 28.1663 | 28.3025 | 28.5727 | 27.8546 |
| Dynamic PSNR | 25.1812 | 24.8071 | 26.5556 | 25.3992 |
| Dynamic alpha | .98842 | .98829 | .99047 | .98676 |

I2 meets the dynamic-PSNR and alpha numerical guardrails but fails the required
visual integrity condition. Actual review of all nine rows of fixed86 and
fixed188 four-model comparison sheets found persistent sheets, detached
support and missing geometry. Fixed86 t146/172/200 still stretches into thin
pieces; fixed188 t43/62/86 still lacks solid bodies. Some regions become
smoother or rounder, which does not establish a complete or correct object.
This is a negative result, not an accepted model or paper reproduction.

The six-camera, nine-time screen was rendered at native1600 resolution. The
remaining four-camera sheets were generated but were not manually reviewed in
this decision; the explicit failure cases already reject the candidate. The
full 6×200 acceptance scan and test-RGB evaluation were therefore not run.

Interpretation: sharing a motion field cannot by itself repair the inherited
canonical geometry. The next investigation qualifies cached, training-only
learned depth as an additional geometric prior, inspired by MoDGS
<https://arxiv.org/html/2406.00434v4>. It must retain matched controls and must
not label pseudo-depth or unseen-view renders as independent ground truth.
