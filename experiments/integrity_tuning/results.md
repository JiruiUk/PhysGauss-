# Completeness-first C1 tuning

Status: completed | Active: H2_temporal_cohesion / metrics_test
Active iteration: 50000 / 50000 | Updated UTC: 2026-11-01T02:15:16.828017+00:00

C1 baseline weights, metrics and renders reused. No baseline retraining.
Eight-hour deadline includes waiting for the current Object-ID queue; GPU work is sequential.
Four20k screens, at most two50k promotions.120 train /30 validation /50 final test.
Only training RGB/alpha supervise losses. No instance IDs or learned material segmentation.
Completeness metrics are2D proxies; they do not prove watertight3D geometry.

| Model | State | Iter | Validation PSNR | Integrity score (lower) | Core low90 | BG spill |
|---|---|---:|---:|---:|---:|---:|
| H1_balanced_coverage | screen_complete | 20000/50000 | 35.4663028717041 | 0.007431643396873023 | 0.0014234624323888126 | 0.015053751150940892 |
| H2_temporal_cohesion | completed | 50000/50000 | 39.283321698506676 | 0.003972160060373721 | 0.000792338244888957 | 0.011724610797872652 |
| H3_combined | completed | 50000/50000 | 40.37668075561523 | 0.003561218916094816 | 0.0008101252275341623 | 0.008109228660232061 |
| H4_color_density | screen_complete | 20000/50000 | 35.61568234761556 | 0.008039857470462498 | 0.002501530001140523 | 0.016135239088695793 |
| H5_weaker_combined | optional_retry | 0/50000 | - | - | - | - |

[Diagnosis](diagnosis.md) | [Monitor](monitor.html) | [Plan](plan.json)

Deadline UTC: 2026-11-01T05:17:35+00:00
