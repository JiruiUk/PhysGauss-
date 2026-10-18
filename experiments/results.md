# PIDG constitutive plan results

Updated: 2026-10-18T22:03:13.771263+01:00

M0 reused five completed original baselines. Active long scope: Rubber Duck only, 50k steps, width1600, seed0. Short controlled data are reported separately.

## Table 1: selected official compatibility benchmark

| Method | Status | Checkpoint | PSNR | Delta PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|---:|
| A0 original (reused) | completed | 50000 | 27.25678 | 0 | 0.97655 | 0.02962 |
| A1 | completed | 50000 | 28.149500 | 0.892719 | 0.976896 | 0.031075 |
| A2a | completed | 50000 | 27.128334 | -0.128447 | 0.976362 | 0.031909 |
| A2b | completed | 50000 | 28.408968 | 1.152187 | 0.976817 | 0.030052 |
| A4_single | completed | 50000 | 28.329363 | 1.072582 | 0.976372 | 0.030201 |
| A6_single | queued | — | — | — | — | — |

A3 official multi-material comparison awaits known labels. A4_single/A6_single use a single shared material; they are explicitly compatible subsets, not real multi-material results. A5 diagnostics reuse A4 training. Full multi-material/interface modules are validated on manufactured data below. D-NeRF/HyperNeRF metrics remain unavailable, not zero. Historical A0 runtime hashes were not recorded.

## Table 2: controlled engineering ablation

All rows below use manufactured two-region data, 300 steps, width400, seed0; these are not 50k official benchmark gains.

| Method | Status | PSNR | SSIM | LPIPS | CMR sampled loss | Constitutive raw | Invalid F | Interface proxy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0_controlled | completed | 28.648554 | 0.975908 | 0.035855 | — | — | — | — |
| A1_controlled | completed | 27.810379 | 0.973001 | 0.037685 | — | — | 0.000000 | — |
| A2a_controlled | completed | 28.364449 | 0.974396 | 0.037367 | 0.000959 | 0.345267 | 0.000000 | — |
| A2b_controlled | completed | 27.745901 | 0.973002 | 0.038171 | 0.001004 | 0.349443 | 0.000000 | — |
| A3_interface_controlled | completed | 28.426605 | 0.975397 | 0.035195 | 0.001063 | 0.446898 | 0.000000 | 0.009364 |
| A4_controlled | completed | 28.046354 | 0.973748 | 0.037500 | 0.001058 | 0.401398 | 0.000000 | 0.009270 |
| A5_controlled | completed | 27.859320 | 0.973501 | 0.036691 | 0.001109 | 0.459185 | 0.000000 | 0.009430 |
| A6_controlled | completed | 27.988436 | 0.970826 | 0.043638 | 0.002045 | 0.429917 | 0.000000 | 0.008697 |

CMR retains the released sample scaling and time convention. Interface values use a canonical centre-line normal proxy. New physical parameters are not claimed recovered. Future-time images use the test-based snapshot protocol, so no leakage-free forecasting claim is made.

## Table 3: manufactured stiffness/force ambiguity

Analytic solutions to a manufactured balance; not material recovery from official images.

| Case | Data rank | Objective rank | Condition | Min SV | mu error | Force error | Max correlation |
|---|---:|---:|---:|---:|---:|---:|---:|
| force_known | 2 | 2 | 1.5000000000000002 | 8.000000 | 0.000000 | 0.000000 | — |
| stiffness_known | 1 | 1 | 1.0 | 4.000000 | 0.000000 | 0.000000 | — |
| both_unknown | 2 | 2 | inf | 0.000000 | 0.750000 | 1.500000 | 0.999978 |
| both_unknown_with_prior | 2 | 3 | 20.549255459247757 | 0.583963 | 0.250000 | 0.500000 | 0.984726 |

A prior can improve objective conditioning while the data remain ambiguous. Actual post-fit Jacobians condition on fixed neural fields and are not proof of global identifiability. The earlier controlled A5 smoke retained about 20% material error and 100% force error despite full conditional rank.
