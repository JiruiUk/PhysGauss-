# C1 object-completeness diagnosis

Existing30 validation images only; no baseline training or GPU rendering repeated.
Alpha holes, disconnected image regions and opaque appearance errors are measured separately.
No semantic segmentation target or ground-truth3D surface is introduced.

| Metric (fractions, lower is better except IoU) | Value |
|---|---:|
| core_low_50 | 0.00014473 |
| core_low_90 | 0.00101670 |
| foreground_missing_50 | 0.00259443 |
| background_false_coverage | 0.01003471 |
| fragment_loss | 0.00692630 |
| extra_fragments | 0.00833333 |
| opaque_appearance_error | 0.00586822 |
| foreground_RGB_L1 | 0.01029228 |
| mask_iou | 0.91621723 |
| integrity_score | 0.00408167 |
| worst_score | 0.01478561 |
| p95_score | 0.00854751 |

Worst validation frames: 0078, 0104, 0098, 0144, 0058

Foreground/GT-component metrics are2D proxies, not proof of watertight3D objects.
Appearance errors at high alpha must not be reported as opacity holes.

Diagnostics: F:\A_Edinburgh\PhysGauss\experiments\integrity_tuning\diagnosis_images

Baseline: F:\A_Edinburgh\PhysGauss\runs\pidg_color_repair\validation_previews\C1_repaired_50000
