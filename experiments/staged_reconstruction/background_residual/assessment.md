# I10: neutralize frozen-background RGB residuals

Both arms actually completed2000 native1600 updates from exact S3-U final6k,
including200 original-physics steps. Full-state/sampler/RNG audit passes.
The source freeze, baseline tensors, Adam groups and frozen S1 are unchanged.

| Model | Actual-GT val dynamic PSNR | Dynamic alpha | Static-region PSNR |
|---|---:|---:|---:|
|S3U_parent|25.181221|0.988421|34.999955|
|I10_control|26.478602|0.989817|35.905383|
|I10_static_residual|26.167859|0.989314|35.253027|

Candidate minus matched control dynamic PSNR: -0.310743dB.
These metrics use actual validation RGB, never the neutralized training target.
No test RGB is used. Direct foreground/known-empty targets and original physics
were retained; SSIM neighborhoods still couple adjacent image regions.

The separate control_repeat_audit.json records a substantial difference
between historical I0 and this control despite exact initial model/Adam/RNG
and camera schedules. Single-pair PSNR differences are not stable causal
estimates; no best repeat is substituted for the preregistered control.

This intervention changes continuation-time RGB supervision only. It cannot
rule out static-derived initialization or damage accumulated before final6k,
and does not establish static geometry truth or a causal percentage.

Original6x9 fixed-camera native RGB/dynamic RGB/alpha images have been rendered.
Actual visual review is recorded separately in visual_review.json; numerical
checks do not prove completeness. The reviewed key probes retain missing
surfaces/fragments, including dynamic-only renders. The candidate fails
completeness and is not promoted. No6x200 acceptance pass is claimed.

Reproduction: the opt-in background_residual prepare/preflight/train/evaluate
modules and scripts.finalize_background_residual. Preparation and evaluation
preserve prior receipts; repeats require a new versioned experiment.
