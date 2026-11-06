# C1 / fixed-E / learned-E total50k comparison

Status: completed | Active: P2_learn_E / metrics_test

Three matched full3k branches, total50k each;120 train /30 validation /50 test. Historical test exposure persists.
P0: C1 control. P1: same C1 + fixed E=1. P2: same constitutive objective + learned shared E.
Fixed Poisson0.3; E bounded[0.05,20]. Global effective prior, no true material labels, no calibrated Pa.
Mean constitutive weight1e-4, interval100 with compensation; start3k/ramp2k. Prior1e-3 on log(E/E0)^2.
Only validation selects snapshots; final test opens after all three finish and selections lock.

| Model | Iteration | Latest val PSNR | E | Test PSNR | SSIM | LPIPS | MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0_C1_control | 50000/50000 | 39.40651257832845 | - | 40.0372620 | 0.9945438 | 0.0104763 | 0.9966279 |
| P1_fixed_E | 50000/50000 | 36.84883003234863 | 1.0000000000000002 | 37.2739143 | 0.9919317 | 0.0155555 | 0.9944728 |
| P2_learn_E | 50000/50000 | 36.890815862019856 | 0.9934803617060427 | 37.2672768 | 0.9920302 | 0.0152616 | 0.9943668 |

Final50k deltas versus P0:
P1_fixed_E: {"PSNR": -2.763347625732422, "SSIM": -0.0026121139526367188, "LPIPS": 0.0050792573019862175, "MS-SSIM": -0.0021550655364990234}
P2_learn_E: {"PSNR": -2.7699851989746094, "SSIM": -0.002513587474822998, "LPIPS": 0.0047853076830506325, "MS-SSIM": -0.002261042594909668}

[5-second monitor](monitor.html) | [Plan](plan.json) | [Preflight](preflight.json)

GT/render galleries: F:\A_Edinburgh\PhysGauss\runs\pidg_c1_material\render_comparisons

Conditional E profiles freeze neural fields; they do not prove data identifiability.
Updated UTC: 2026-10-29T20:16:34.902968+00:00
