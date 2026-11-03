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

## Completed control milestone

Unchanged-row arm completed18k from full12k,3273 points. Audit passed:
all six Adam counters18k,6000 camera samples replayed from parent,6 absolute
confidence refreshes and frozen source/input/history checks. Checkpoint SHA256:
`b0c5e25b8b3da9013fd745e399f02a9305fa678b49b6d778dd6750ab0701b584`.
Visibility arm is running; no final quality conclusion before matched evaluation.

## Completed visibility milestone

Visibility arm also completed18k,3031 retained points and242 reversible
quarantined rows. All six Adam counters18k;6000 parent-continuation camera
samples,6 confidence refreshes and frozen source/history checks passed.
Checkpoint SHA256: `883101f36ab668ea6b9d6fb3a5616422ca54b2067f97a50f4d2f220332bcb015`.
Matched native evaluation is running. No promotion before actual images.

## Final C1 assessment

Both arms and exports completed. Manual native review found no clear cleanup;
visibility not promoted. Frozen CT val25.7836/control vs25.7377/visibility;
new val22.0890 vs22.0206. Three videos decode200 frames, PLY exact.
See v1/assessment.md. Synthetic RGB admission audit is now running separately;
next single candidate is nuisance separation, not stronger deletion.

## Independent admission audit completed

v1 sparse calibration gate failed0/24; retained. v2 has96 train RGB/camera/time
per case, all six-source gates passed before inference. Independent static GT
scores: precision81.9%→86.1%, surface coverage about99.4%, thin coverage100%.
Only RGB-to-geometry admission has been tested, not complete Gaussian training.
7 mechanism tests and source/train/evaluator-file hash checks passed.
See synthetic_v2/assessment.md. Next is one per-view nuisance separation pilot,
first with full Gaussian synthetic validation; do not repeat C1 continuation.

## Next authorized stage is concretely bounded

See experiments/staged_reconstruction/S1_nuisance/plan.md and protocol.json.
No nuisance implementation/full Gaussian synthetic training has started yet.
Complete tests/native recovery and both synthetic-case gates first, then one
matched real18k→24k control/nuisance continuation. Never restart C1 simply
because its pipeline state still names pending review; final_review.json and
assessment.md already complete that review. Automatic follow-up remains active
for this new cleanup scope; oldpidg-s1-s6 remains paused.
