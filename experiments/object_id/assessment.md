# Object ID experiment assessment

Matched C1, frozen identity and learned identity; total50k each, same full3k prefix.
C1 uses previously completed50k weights/metrics/renders; new arms use the same50 test frames after selection lock.

| Model minus reference | PSNR delta | SSIM delta | LPIPS delta | MS-SSIM delta | PSNR better frames |
|---|---:|---:|---:|---:|---:|
| O1_frozen_ID minus O0_C1_control | -1.5613022 | -0.0010020 | +0.0012525 | -0.0004367 | 12/50 |
| O2_selfsup_ID minus O0_C1_control | -1.8912544 | -0.0013879 | +0.0015294 | -0.0010083 | 6/50 |
| O2_selfsup_ID minus O1_frozen_ID | -0.3299522 | -0.0003859 | +0.0002770 | -0.0005716 | 23/50 |

O2 vs O0 measures the combined object-motion package; O2 vs O1 probes learned membership against frozen geometric membership.
Image changes alone do not establish correct physical object IDs. Check identity renders, occupancy, gradient and shuffled-identity diagnostics.
C1 foreground alpha supervision remains; only the added identity module is instance-label-free.
No nonpenetration or material-identification claim.

RGB/GT gallery: F:\A_Edinburgh\PhysGauss\runs\pidg_object_id\render_exports\final50000\gt_c1_frozen_learned\index.html
Identity maps: F:\A_Edinburgh\PhysGauss\runs\pidg_object_id\identity_exports\index.html
