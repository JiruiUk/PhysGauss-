# L7-J: cross-frame scale and wider RGB-anchor coverage

User authorizes testing cross-frame joint calibration and broader anchors before
deciding whether weak depth is usable. This is one default-off read-only audit,
not Gaussian training. J6 and fixed Small raw predictions from L7-P stay frozen.
No new color classes, alpha selectors, static labels, counts or GT are inputs.

Expand only stationary-compatible RGB hypotheses: full-image tiled Shi-Tomasi
features followed by fixed pyramidal LK, forward and reverse sequential tracks.
Retain every seed and failure. Use original texture/NCC/closure/direction and
stationary triangulation thresholds. LK success is not learned visibility or GT.
Original T4 motion hypotheses and eligibility are never replaced by these tracks.

Avoid repeated observations masquerading as independent scale evidence. Merge
stationary tracks with overlapping >=2 frames and <=1px separation in every
overlap. Whole connected groups are fit/check assigned by SHA256 of lowest ID;
two fit halves are group-disjoint too. Per frame deduplicate4px; where a held
cell exists discard fit members of that cell, retaining held. The same group
never fits and checks scale in different frames. Groups are hypotheses, not IDs
of physical objects. Report any large merged group and all removed duplicates.

Normalize each frozen relative map by its full-image median and q90-q10; no
pixel classification. Jointly fit120 positive slopes and120 offsets to FIT
inverse depths with relative residuals and fixed Huber loss. A fixed quadratic
second-time-difference penalty on the normalized slope/offset (scaled by median
fit inverse depth) couples adjacent frames. Both disjoint fit subsets are fit
independently with the identical algorithm. No GT, held anchor or dynamic parent
depth enters any fit. This smoothness is a prior; it cannot prove scale accuracy.

Keep L7-P local counts, spatial extent, inverse spread and5%/10% held/split
checks. Current-frame fit hull is mandatory: neighboring frames cannot extend
the spatial training support. A same-expanded-anchor framewise fit is a declared
diagnostic control only; old L7-P is a frozen baseline, not rerun. Report each
control's full denominator, 4x4 occupied-cell and hull area coverage, and native
dynamic observations inside these hulls. No result-selected estimator/weight.

Reuse L7-P motion/backprojection/boundary tests unchanged,122 eligible native
tracks from222 candidates, all4433x12 boundary observations. Original local gate
and old T4 global failures remain unchanged. Missing evidence remains unknown.
If gate fails, no training; if passed, separately preregister native qualification
and matched full-state weak-depth training before any optimizer update.

Freeze protocol/source/input hashes and commit before new anchors/decisions.
CPU deterministic extraction, independent fixed0/52/108 clip replay, independent
all120 fit/residual/decision verification, default-off/unit tests, reviewed
fixed0/24/72/119 panels and full temporal coverage. Preserve failures separately.
