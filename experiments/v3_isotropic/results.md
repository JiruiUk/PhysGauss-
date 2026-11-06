# V3 + isotropic regularization

V3 parent; additional50k; 120 train/30 validation/50 test; shared frame clock; iso=0 vs .001

Status: saved_stopped
Active: V3_continue_control / long

Both arms inherit the same completed V3 validation-selected weights. The first1000 steps share a full checkpoint; iso starts at1001.
Validation alone selects the four best snapshots. After both arms finish, lock selections and evaluate final50k plus four validation-selected snapshots on test.

| Model | Extra iter | Latest validation PSNR |
|---|---:|---:|
| V3_continue_control | 10489/50000 | 39.12321294148763 |
| V3_iso_1e3 | 1000/50000 | 37.564193662007646 |

## Fixed-step final test results

| Model | PSNR | SSIM | LPIPS | MS-SSIM |
|---|---:|---:|---:|---:|
| Previous V3 (before extra training) | 40.0917091 | 0.9946094 | 0.0103923 | 0.9962949 |

## Validation-selected snapshots: final test metrics

| Model | Validation objective | Extra iter | PSNR | SSIM | LPIPS | MS-SSIM |
|---|---|---:|---:|---:|---:|---:|

The previous V3 test set has historical exposure. This prospective validation-only protocol cannot erase it.
[5-second chart monitor](monitor.html) | [Plan](plan.json) | [GPU preflight](preflight.json)

GT/render pairs absolute directory: F:\A_Edinburgh\PhysGauss\runs\pidg_v3_isotropic\render_comparisons

Updated UTC: 2026-10-26T14:25:31.322270+00:00
