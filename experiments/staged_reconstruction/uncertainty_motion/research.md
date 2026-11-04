# USplat4D-informed reliability qualification for S3-U

The target remains actual complete objects in the previously fixed novel-view
screen and full sequences. No candidate has passed that target. I10 did not
restore geometry, and prior uniform trajectory neighborhoods (I1) did not
establish reliable point identities. The next intervention must distinguish
observed anchors from uncertain points rather than repeat uniform strain.

## Primary sources, pinned and read

- Paper: <https://arxiv.org/html/2510.12768v3>, sections4.1--4.3 and appendixB.
- Official project: <https://tamu-visual-ai.github.io/usplat4d/>.
- Official repository: <https://github.com/TAMU-Visual-AI/usplat4d>, commit
  `c7a55211395c4020fc28fd61a09b172abf773710`.
- Its `external/gsplat` gitlink pins <https://github.com/percool/gsplat>, commit
  `2486a4ad5343a4e1a6f57fcdd2845c17577816d2`.

Downloaded source text is never imported or executed. `official_source.json`
and `backend_source.json` identify each SHA256 and upstream Git blob SHA1;
the full upstream trees are retained. Fetch helpers verify the Git blob hash,
verify existing files before reusing them, keep TLS verification, and do not
install dependencies. The source snapshot is an audit dependency, not a new
renderer used for PIDG training.

The official gsplat `RasterizeToPixels3DGSFwd.cu:162--196` computes `vis=alpha*T`,
accumulates `vis*vis`, `vis`, and a usage count per Gaussian, then updates T.
`lib_usplat4d_prep/usplat4d_prep.py:1894--1956` consumes squared support and
gates points by projected foreground membership, RGB MAE, SSIM and prior-depth
gradient. This center-projection filter differs from the paper's all-covered-
pixels convergence indicator. `main2` selects high-contribution candidates
per voxel and retains recurring support for at least5 frames. The code has
normalization and count caps; it is not an exact top2% implementation throughout.

At `main3:974--976`, per-point depth information is overwritten by a user-supplied
constant `depth_ratio`. The key-position objective uses a rotated diagonal
quadratic metric (`I0[...,2,2]=depth_ratio`, lines267--284); reducing that entry
reduces the depth penalty. Do not copy an uncertainty/covariance label into a
precision loss without checking this distinction. No depth anisotropy or new
depth targets are adopted in this diagnostic.

## PIDG adaptation and its limits

The unchanged PIDG renderer RGB operator is linear in independent override
colors for fixed geometry: `I=A*c+b`. Its exported per-pixel Gaussian buffer
cannot replace A: it has only20 entries per pixel and writes `alpha*T` after
updating T, unlike RGB accumulation. Baseline source files are preserved.

The new independent diagnostic uses native color backward. For independent
Rademacher pixel signs z, `E[(A^T z)^2]=diag(A^T A)`. Three RGB channels supply
three independent probes per backward. It uses private local RNG, detached
geometry, no optimizer, and never writes model `.grad`. Exact linear VJPs also
measure the fraction of a point's compositing mass that lies in converged
foreground. This is a numerical adaptation, not full USplat4D reproduction.

Qualification proceeds from analytic synthetic operators to explicit one-hot
native renders, then to original training observations. An initial128 RGB-batch
budget failed the predeclared10% median error gate. The versioned convergence
audit keeps that failure and evaluates512/2048 batches under identical gates.
Two diagnostic wrapper errors (empty-slot -1 handling and detaching exported
weights for serialization) are retained in logs/protocol lineage. Neither
changes the baseline or measurement formula.

The observed pilot is fixed before rendering:6 train frames at width800,
one repeated frame at native1600, two independent seeds, exact one-hot checks
on48 population-spanning rows, and stability of qualified top2% rankings.
All120 train RGB hashes are locked before this pilot; no val/test RGB is read.
The pilot's foreground/error-mass gate is explicitly our adaptation and does
not reproduce the official depth-gradient/SSIM filter. Passing measurement
qualification does not certify accurate depth, same-object association, full
surface coverage, or cross-occlusion physical identity.

Only a qualified all-training-time audit can propose recurring anchors and
spatial coverage. No new motion loss, graph training, material/force module,
object count, or learned segmentation has been integrated. Subsequent training
requires its own opt-in protocol, tests, native flags-off check, actual short
training/recovery checks and matched controls. Existing appearance/static gates
and fixed6x9 then6x200 actual visual acceptance remain unchanged. Numerical
repeat sensitivity from I10 must remain visible when interpreting comparisons.

## Independent code mapping

Existing baseline mapping remains in `PIDG_REPO_MAP.md`; unrelated edits there
are preserved. This additive diagnostic lives in
`variants/staged_reconstruction/uncertainty_motion/`: `core.py` owns color-VJP
support and observation-qualified recurrence, `native_audit.py` validates the
unaltered renderer on explicit small populations, and `observed_audit.py`
audits the exact S3-U parent on train RGB. No original PIDG training entry,
field, renderer, dataset loader, checkpoint or evaluation file is modified.
