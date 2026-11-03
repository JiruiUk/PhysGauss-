# Wider-pair known-camera pointmaps: qualification results

The official frozen MonST3R float32 model actually ran all12 new train-only
pairs. Both time-specific pointmaps are retained. Synthetic recovery tests:
**10 passed**. No PIDG optimization or depth targets were produced.

| Anchor | Partners | Dynamic confidence coverage | Static support | Raw first projection median/p90 (px) |
|---|---|---|---|---|
|2|10/20|0.0% / 0.0%|0/5537|unavailable; unavailable|
|43|34/52|0.0% / 0.0%|0/512|unavailable; unavailable|
|86|76/95|1.2% / 0.0%|3074/0|56.823/58.833; unavailable|
|146|136/155|0.0% / 22.2%|926/3891|unavailable; 26.411/29.505|
|188|179/170|57.4% / 80.5%|10970/10625|21.662/45.507; 28.743/54.866|
|200|190/182|100.0% / 57.4%|11147/11028|31.301/62.250; 23.906/53.692|

| Anchor/partner | Static scale | Camera scale | Camera held support | Camera held projection median/p90 (px) |
|---|---|---|---|---|
|2/10|unavailable|unavailable|0|unavailable|
|2/20|14.648|4.745|3993|111.989/136.515|
|43/34|unavailable|unavailable|0|unavailable|
|43/52|14.628|unavailable|21|unavailable|
|86/76|13.180|unavailable|0|unavailable|
|86/95|unavailable|unavailable|0|unavailable|
|146/136|9.926|unavailable|0|unavailable|
|146/155|10.183|9.371|2012|23.225/36.213|
|188/179|12.550|6.023|8188|52.301/79.001|
|188/170|12.876|10.256|7662|52.603/73.410|
|200/190|13.054|0.258|8991|653.180/742.911|
|200/182|12.657|6.959|7898|46.917/72.230|

wide_pairs_static_scale_control: 0/12 pairs and
0/6 anchors pass their respective checks.
Entire arm passed: False.

wide_pairs_known_camera_scale: 0/12 pairs and
0/6 anchors pass their respective checks.
Entire arm passed: False.

The original adjacent-pair control had 0/12 passing pairs.
The wide-pair control changes pair selection only. The second arm changes
the scale estimator on exactly those same predictions. Scalar alignment
cannot change first-view point-to-pixel projection or raw confidence.
It also cannot restore an unseen surface absent from a pointmap.

These are necessary consistency checks, not synchronized novel-view GT.
S1 depth remains a model-conditioned reference. Failure of this pilot is
not rejection of the whole paper or evidence for an inaccurate static S1.
In particular, known-camera projection/confidence failures are independent
of fitting to static reference depth. Original gates are unchanged.

Visual panel masks show all raw-confidence>=3 pixels, including predictions
on the black input background; these are not foreground geometry targets.
The scalar estimate has no camera/model optimization or pointmap warp.

The evidence figure is generated; actual visual inspection is recorded
separately in visual_review.json. Missing or nonfinite statistics are null
and fail applicable gates. No full physics training follows a failed gate.

Reproduce in separate processes (each requires the explicit opt-in flag):
`python -m variants.staged_reconstruction.calibrated_pointmaps.prepare`,
`python -m variants.staged_reconstruction.calibrated_pointmaps.infer`,
`python -m variants.staged_reconstruction.calibrated_pointmaps.qualify`.
Flag: `--enable-calibrated-pointmap-pilot`. Existing receipts are preserved;
a repeat requires a new versioned experiment. Reporting command:
`python -m scripts.finalize_calibrated_pointmaps`.
