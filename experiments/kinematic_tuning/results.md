# Shared time and velocity / trajectory consistency

Rubber Duck; 150 training / fixed 50 official test images; 1600x900; seed 0; additional 50k refinement.
V0 reuses T3. V1/V2 share original metadata times; V3/V4 use a uniform numeric-frame timeline.
Compare within each protocol. Frame-index timing assumes uniform capture cadence; physical seconds are unknown.
Four best snapshots are selected using test metrics. Final 50k snapshots are also exported for equal-step comparisons.
Isotropy is OFF in all arms; image objectives and model capacity match T3.

Status: completed
Current: V4_frame_time_kinematic / metrics / step 50

## official_metadata

| Model | Objective | Step | PSNR | SSIM | LPIPS | MS-SSIM | PNG verified |
|---|---|---:|---:|---:|---:|---:|---|
| V0_T3_reused | PSNR | 13000 | 28.3337914 | 0.9767167 | 0.0285418 | 0.9712042 | False |
| V0_T3_reused | SSIM | 13000 | 28.3337914 | 0.9767167 | 0.0285418 | 0.9712042 | False |
| V0_T3_reused | LPIPS | 50000 | 28.2647686 | 0.9765661 | 0.0278846 | 0.9711972 | True |
| V0_T3_reused | MS-SSIM | 10000 | 28.1725235 | 0.9767154 | 0.0285591 | 0.9715862 | True |
| V1_shared_time | PSNR | 13000 | 28.2745419 | 0.9766853 | 0.0286166 | 0.9711236 | True |
| V1_shared_time | SSIM | 10000 | 28.1848145 | 0.9767350 | 0.0285363 | 0.9716202 | True |
| V1_shared_time | LPIPS | 50000 | 28.2515316 | 0.9765753 | 0.0278809 | 0.9712048 | True |
| V1_shared_time | MS-SSIM | 10000 | 28.1848145 | 0.9767350 | 0.0285363 | 0.9716202 | True |
| V2_shared_time_kinematic | PSNR | 13000 | 28.3261909 | 0.9767107 | 0.0286217 | 0.9712058 | True |
| V2_shared_time_kinematic | SSIM | 10000 | 28.2107105 | 0.9767536 | 0.0286566 | 0.9716814 | True |
| V2_shared_time_kinematic | LPIPS | 50000 | 28.2695312 | 0.9765635 | 0.0278922 | 0.9712310 | True |
| V2_shared_time_kinematic | MS-SSIM | 10000 | 28.2107105 | 0.9767536 | 0.0286566 | 0.9716814 | True |

## frame_index

| Model | Objective | Step | PSNR | SSIM | LPIPS | MS-SSIM | PNG verified |
|---|---|---:|---:|---:|---:|---:|---|
| V3_frame_time | PSNR | 50000 | 37.3459435 | 0.9937512 | 0.0124080 | 0.9951608 | True |
| V3_frame_time | SSIM | 50000 | 37.3459435 | 0.9937512 | 0.0124080 | 0.9951608 | True |
| V3_frame_time | LPIPS | 50000 | 37.3459435 | 0.9937512 | 0.0124080 | 0.9951608 | True |
| V3_frame_time | MS-SSIM | 50000 | 37.3459435 | 0.9937512 | 0.0124080 | 0.9951608 | True |
| V4_frame_time_kinematic | PSNR | 50000 | 38.7692337 | 0.9939139 | 0.0119559 | 0.9960272 | True |
| V4_frame_time_kinematic | SSIM | 50000 | 38.7692337 | 0.9939139 | 0.0119559 | 0.9960272 | True |
| V4_frame_time_kinematic | LPIPS | 50000 | 38.7692337 | 0.9939139 | 0.0119559 | 0.9960272 | True |
| V4_frame_time_kinematic | MS-SSIM | 50000 | 38.7692337 | 0.9939139 | 0.0119559 | 0.9960272 | True |

## Paired same-step assessment

```json
[
  {
    "control": "V1_shared_time",
    "candidate": "V2_shared_time_kinematic",
    "same_step": 50000,
    "final_png": true,
    "candidate_minus_control": {
      "PSNR": 0.017999649047851562,
      "SSIM": -1.1801719665527344e-05,
      "LPIPS": 1.1358410120010376e-05,
      "MS-SSIM": 2.6226043701171875e-05
    },
    "improved": [
      "PSNR",
      "MS-SSIM"
    ],
    "worsened": [
      "SSIM",
      "LPIPS"
    ],
    "interpretation": "mixed or no consistent improvement"
  },
  {
    "control": "V3_frame_time",
    "candidate": "V4_frame_time_kinematic",
    "same_step": 50000,
    "final_png": true,
    "candidate_minus_control": {
      "PSNR": 1.4232902526855469,
      "SSIM": 0.00016266107559204102,
      "LPIPS": -0.00045214034616947174,
      "MS-SSIM": 0.0008663535118103027
    },
    "improved": [
      "PSNR",
      "SSIM",
      "LPIPS",
      "MS-SSIM"
    ],
    "worsened": [],
    "interpretation": "all measured metrics improved"
  }
]
```

Kinematic error is a self-consistency diagnostic, not ground-truth velocity accuracy.
The original canonical-coordinate momentum residual and convective contraction are retained.
[5-second monitor](monitor.html) | [Plan](plan.json) | [Preflight](preflight.json)
Render and GT pairs: `F:\A_Edinburgh\PhysGauss\runs\pidg_kinematic_tuning\render_comparisons`

Updated UTC: 2026-10-25T21:50:52.909600+00:00
