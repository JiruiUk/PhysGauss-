# H3 worst-frame repair ablations

Status: completed | Active: D3_combined | 50000/50000 iter

H3 reused. D1 motion-compatible local affine regularization; D2 covariance curvature regularization; D3 both.
Training data, RGB/alpha preprocessing, clock, original stack sampler and camera-selection code are unchanged.
Shared random-training3k full states, total50k each;120 train/30 validation/50 fixed test.
No test/validation RGB or alpha supervises these changes. Validation chooses; historical test exposure remains.

| Model | Iter | Validation PSNR | Integrity score |
|---|---:|---:|---:|
| D1_motion_affine | 50000 | 40.34098059336345 | 0.00420780171222502 |
| D2_covariance_continuity | 50000 | 39.34107856750488 | 0.004725416841982586 |
| D3_combined | 50000 | 40.11097081502279 | 0.0035416120776727692 |
