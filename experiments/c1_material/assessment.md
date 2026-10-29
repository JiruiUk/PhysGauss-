# C1 shared-stiffness assessment

All3 runs completed. Primary table: final50k, same50 test images.
Matched shared full3k branch; validation-only selections. Historical test exposure persists.

| Model | PSNR | SSIM | LPIPS | MS-SSIM | Foreground PSNR | Mask IoU | Interior low coverage % |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0_C1_control | 40.037262 | 0.994544 | 0.010476 | 0.996628 | 31.7603 | 0.896682 | 0.005241 |
| P1_fixed_E | 37.273914 | 0.991932 | 0.015556 | 0.994473 | 29.5934 | 0.835099 | 0.007880 |
| P2_learn_E | 37.267277 | 0.992030 | 0.015262 | 0.994367 | 29.8208 | 0.843039 | 0.037469 |

## Observed effects

P1-P0 PSNR -2.76335dB;48/50 frames worse. P2-P0 -2.76999dB;49/50 worse.
Foreground PSNR and Mask IoU also decline. Learning E versus fixed E gives mixed small mean differences.
Learned E:1 ->0.993480 before the final update, about0.652% change;470 scheduled material updates.
P2 largest logged weighted constitutive loss13.34475 and ||dL/dF||187195.078 at iteration28400.
P1 maximum ||dL/dF||1378.368 at49000. These are local F derivatives, not model-weight gradient norms.
The nearby recorded validation PSNR does not show an immediate drop across the P2 event; a single spike is not a causal explanation by itself.

## Judgment

Keep P0 as the image-quality control. Current global effective-material regularization is not a default improvement.
Before another ablation, inspect F admissibility/conditioning, constitutive-gradient tails and interval compensation; validate rest-configuration and material-region assumptions.
Separate these checks from any later material learning-rate or weight sweep. No new training is launched by this assessment.

[GT + P0 + P1 + P2](../../runs/pidg_c1_material/render_exports/final50000/gt_p0_p1_p2/index.html)
[Raw run results](results.md) | [Full assessment and source hashes](assessment.json)
