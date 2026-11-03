# S1-D5: evaluator-only RGB matching measurement audit

D4 has thousands of finite learned-order proposals but the original RGB fit
cost vetoes almost all. The sole accepted change is geometrically worse even
though both held-source costs improve. Do not relax its thresholds or replace
the pretrained model. Before another proposal/training mechanism, audit what
the existing RGB matcher measures at known true geometry.

This is a privileged evaluator, not a reconstruction candidate. Freeze original
D1 UV/depths, six source indices, RGB/cameras and adaptive_static patch_features/
source_cost. For supported reference-visible static rays, compare costs at D1,
continuous GT and the nearest original inverse-depth-grid hypothesis to GT.
Ask whether an oracle correct-depth proposal would survive the ORIGINAL four
fit/best3 and two held checks. GT is not provided to inference, stored as new
depth/normal/mask assets or used to select any training point or loss.

Group diagnostics by reference5x5 window being on one visible GT face versus
crossing a face/visibility boundary, and by analytic true source visibility.
The latter is evaluator-only ray intersection at the continuous projected
world point, with original source image geometry margins; no source label is
exported. These groups disentangle a wrong measurement objective, occlusions,
discrete sampling and a failure of proposal search. They do not establish a
static classifier for actual data.

Also replay three fixed analytic plane controls: the fractional-pixel baseline
fixture that originally exposed bias, the integer-shift control, and one
slanted plane. Keep the same RGB texture,5px patch implementation and192-depth
grid; evaluate the declared curves rather than tuning windows or thresholds.
These are diagnostics of the matcher, not additional model candidates.

Primary literature motivates the question, not a claim that its mechanism is
already our dominant cause. Read PatchMatch Stereo BMVC2011 sections1/2.1/2.2:
slanted support and boundary-mixed windows can bias local matching. ACMM's
official CVPR2019 abstract emphasizes visibility/view selection and geometric
consistency; full CVF PDF was403, so do not claim it was fully read. Neither
method has been implemented or benchmarked here. See literature.md.

Tests and source/input freeze precede this single two-condition audit. Keep
all previous reports/weights/gates and unrelated work. Save aggregate/per-view
statistics and fixed0/24/72 color diagnostic figures only; no per-pixel GT-cost
arrays/point lists/GT normals, inferred model updates or Gaussian training.
Actual image review and plan sync required. No direct promotion from oracle
cost improvements and no claim that RGB data are impossible if they fail.
