# H3 worst-frame repair ablations

Status: saved_stopped | Active: R2_hard_train_frames | 31900/50000 iter

H3 reused. R1 unit covariance rotation; R2 training-error hard-frame sampling; R3 both.
Shared random-training3k full states, total50k each;120 train/30 validation/50 fixed test.
No test/validation RGB or alpha supervises these changes. Validation chooses; historical test exposure remains.

| Model | Iter | Validation PSNR | Integrity score |
|---|---:|---:|---:|
| R1_unit_rotation | 50000 | 40.152028528849286 | 0.003932785008621487 |
| R2_hard_train_frames | 31900 | 38.036101786295575 | 0.004450560312013642 |
| R3_combined | 0 | - | - |
