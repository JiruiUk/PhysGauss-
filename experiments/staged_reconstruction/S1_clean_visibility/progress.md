# S1-C cleanup research and matched continuation

User objective: minimize artifacts while retaining clean, complete structure.
No known static color/area, body count or material labels may enter training.
This is a static-reconstruction improvement; do not claim complete dynamic PIDG
reconstruction or replace the preserved S3-U model.

The literature review is in `literature.md`. Current v1 tests only conditional
ray-evidence quarantine, from full G1 12k to matched 18k. No growth, additional
loss, physics module, weight sweep or object-count estimation is bundled in.
The parent and all rejected hypotheses remain recoverable.

Discovery: 242/3273 parent points quarantined; 3031 retained. Reservoir537/19953
quarantined; geometry support276523 to275086 pixels across all120 train views.
Missing and occluded evidence never supplies a negative vote. These inferred
depths are not independent static/depth GT.

Read-only native review: seven-view montage and native86/188 inspected. Main
column/pedestal surfaces retained; yellow smears and blue spikes still visible.
Frozen inferred validation25.4932 to25.2486dB before retraining. This passes the
predeclared pilot gate but establishes no cleanup success or geometric truth.

55 tests passed:10 new mechanism/state/flags-off checks and45 inherited checks.
Sources/protocol/inputs frozen; full-parent 12000→12040→12080 two-process smoke
and Adam/RNG/sampler audit must pass before production. `v1` contains receipts.

Production command, after smoke and implementation milestone commit:
`python experiments/staged_reconstruction/S1_clean_visibility/run_pipeline.py --enable-clean-static`.
Run with inherited CUDA/MSVC environment and workspace plus `src` on PYTHONPATH.
The pipeline serializes both arms, audit and independent native evaluation.
Check `runs/pidg_staged_reconstruction/S1_clean_visibility/v1/pipeline.json`,
`state.json`, actual process/logs, and both lock files before launching anything.
`stop.request` saves state and releases GPU; never auto-resume user stops.

Required completion: per-arm full-state audits and separate milestone commits,
same frozen proxy masks, actual native86/188 and early/late images, exact/finite
PLY, decoded200-frame original-trajectory and fixed-view videos, manual review,
and honest assessment. Do not promote from point count, PSNR or zero static
time variation alone. Preserve failed attempts and prior models.

Next: independent RGB synthetic audit with evaluator-only GT and independent
camera/time variation. Then choose one bounded surface-constraint or nuisance
appearance-separation experiment from evidence, as described in the literature
plan. Do not start competing full candidates or resume old S1–S6 automation.
