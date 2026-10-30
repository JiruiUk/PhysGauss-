# C1 fixed object-center grouping review

This is an offline review milestone, not a trained object-center variant.
The frozen `C1_repaired` final-50000 checkpoint is read without modification.
No changes to the PIDG/C1 training or inference path are installed.

## Reproduce

```powershell
. ./reproduction/scripts/enable_cuda.ps1 -ProjectRoot (Get-Location).Path
& ./envs/pidg-cu128/python.exe -m pytest tests/test_object_centers.py -q
& ./envs/pidg-cu128/python.exe -u -m variants.object_centers.audit
& ./envs/pidg-cu128/python.exe -u -m variants.object_centers.audit --only-motion
```

The first command generates canonical K=5 and trajectory K=5,6,8 candidates.
The last command appends motion-dominant K=5,6 candidates. Existing artifacts
from the first four candidates are retained.

The gallery is
`runs/pidg_object_centers/grouping_review/c1_final50000/index.html`.
Each candidate has 12 GT | C1 RGB | group-color PNGs, fixed kernel IDs,
weighted 3D center trajectories, and a trajectory figure. The same label stays
attached to each kernel across all sampled times. Gray means unassigned.

## Evidence and features

Only predetermined **training** frames 2,19,36,55,74,92,108,127,146,163,182,200
are loaded. The audit creates its own loader with an empty test split; no
validation/test RGB contributes to clustering. Scene time is `(frame_id-1)/199`.

Eligibility requires opacity >=0.03 and foreground-contributor-candidate votes
in at least two of the 12 views. A vote means a kernel index appeared among
the renderer's stored footprint candidates at a pixel with GT alpha >0.95
and rendered alpha >0.1. This is not an exact accumulated contribution weight,
nor proof that the kernel belongs to the visible object's surface.

Weights are frozen `opacity * max(votes,1)`. K-means is deterministic with seed0.
Let s be the norm of the reference XYZ 5th-to-95th percentile bounding diagonal,
and D the concatenated 12-time displacement from the first sampled pose,
divided by sqrt(12). Features are:

- canonical: `X/s`;
- trajectory: `[x(t_first)/s, 2 D/s, 0.75 mean_RGB]`;
- motion dominant: `[0.15 x(t_first)/s, 4 D/s, 0.75 mean_RGB]`.

RGB is existing C1 predicted SH appearance, averaged over the sampled views.
None of these features provide actual object-instance IDs. No learned material
segmentation or per-kernel stiffness/force parameter is introduced.

## Review finding

Of 86,143 kernels, 48,485 are eligible and 37,658 remain unassigned. The
unassigned percentage is a fraction of kernel count, **not** image hole area.
Inspection of early and late renders finds merged ducks, split ducks, and
blue geometry assigned to multiple groups. Increasing K and emphasizing motion
does not establish a one-to-one correspondence with physical objects.
Object count is not inferred reliably by choosing K=5 or K=6.

These are candidate groups for inspection. They have **not** been accepted as
contact labels. A cross-group repulsion loss could separate parts of one duck,
while two ducks sharing a group would receive no mutual contact penalty.
Group render colors replace learned radiance, so dark/hidden kernels can become
visible and the panel can reveal geometry that is inconspicuous in C1 RGB.

Before training object centers/contact, obtain consistent instance membership
(e.g. reviewed training-frame object masks with temporal identity propagated
to kernels), retain uncertain points as unknown, and inspect contact frames.
This needs separate review; semantic membership cannot be asserted from these
fixed clusters alone.

## Standalone tested primitives and intended later integration

`model.py` implements weighted group centers, zero-mean local residuals, a
piecewise-linear trainable center trajectory, topology ID inheritance, and a
soft cross-group effective-sphere overlap proxy. They are **not connected to
the C1 optimizer or renderer**. Center knots in this review are measured from
existing C1 predictions, not newly learned motion.

For an accepted assignment g_i, frozen positive weights w_i define

`c_g(t) = sum_{i:g_i=g} w_i x_i(t) / sum_{i:g_i=g} w_i`,
`r_i(t) = x_i(t) - c_{g_i}(t)`.

An eventual opt-in model can learn the center trend plus a weighted-zero-mean
local deformation, removing ambiguity between global and local translation.
Unknown kernels must retain the original path. Densification children inherit
parent IDs and pruning must apply the identical topology mask to the ID bank.
Snapshot-specific IDs cannot be transferred blindly to a different kernel
ordering or a new random initialization.

These are opacity/evidence-weighted geometric centers, **not physical centers
of mass**. Gaussian tails overlap normally. The supplied sphere penalty is a
soft proxy on sampled cross-group pairs, not a guarantee of nonpenetration;
it can penalize legitimate surface contact and needs calibration before use.
No long training or PSNR improvement is claimed at this stage.

## Follow-up: three motion-only methods

```powershell
. ./reproduction/scripts/enable_cuda.ps1 -ProjectRoot (Get-Location).Path
& ./envs/pidg-cu128/python.exe -m pytest tests/test_motion_grouping.py tests/test_object_centers.py -q
& ./envs/pidg-cu128/python.exe -u -m variants.object_centers.motion_review
```

`motion_grouping.py` adds separately callable displacement-only and
displacement-plus-trajectory-derived-velocity features. Neither feature uses
RGB or absolute XYZ. Actual nonuniform training time intervals define velocity.
Each feature block uses a shared robust amplitude scale, retaining differences
between kernel motion amplitudes. Temporal quadrature corrects unequal spacing.

The third method refines the second method's clusters with a persistent spatial
graph: reference-pose 16NN pairs within the same base motion group; distance
below `min(0.08, 2.5*min(local16NN radii))` in at least80% of the12 sampled poses.
Connected components with fewer than128 kernels remain unknown. It splits
clusters, so final component counts can exceed base K. Short-lived contact does
not automatically create a retained graph edge, but continuous touching or
incorrect base motion splits can still produce wrong instance groups.

`motion_review.py` reuses the checksum-verified old training evidence and source
checkpoint, compares baseK=5/6 for each method, and verifies new GPU poses and C1
RGB against the cache. It writes a separate gallery under
`runs/pidg_object_centers/grouping_review/motion_methods_final50000`, with72
triptychs and24 GT/C1/T1/T2/T3 comparison PNGs. Previous candidate images and
receipts are preserved; their gallery receives a navigation link.

Spatial refinement yields19/18 components and modest visual separation of
initially merged upper ducks. The reviewed later frames still have fragments
and mixed duck/obstacle groups; no candidate is accepted as physical object
identity. See `experiments/object_centers/motion_grouping_review.md`.

## No requested group count

```powershell
. ./reproduction/scripts/enable_cuda.ps1 -ProjectRoot (Get-Location).Path
& ./envs/pidg-cu128/python.exe -m pytest tests/test_adaptive_grouping.py tests/test_motion_grouping.py tests/test_object_centers.py -q
& ./envs/pidg-cu128/python.exe -u -m variants.object_centers.adaptive_review
```

`adaptive_grouping.py` contains **no K-means, target K, or base group labels**:

- N1: HDBSCAN on displacement trajectories;
- N2: HDBSCAN on displacement plus trajectory finite-difference velocity;
- N3: a direct motion/spatial graph whose connected components define groups.

N1/N2 use installed scikit-learn1.8.0 HDBSCAN with min_samples16, EOM selection,
allow_single_cluster=True, and minimum cluster sizes128/256. One group is
allowed; noisy points stay unknown, without nearest-center reassignment. These
are density parameters, not a requested object count. Density is unweighted
kernel count because this sklearn API has no fit sample_weight; the original
opacity/vote weights are retained only for center/dispersion statistics.
Exported membership strength is not calibrated instance correctness.

N3 does not start from either K-means or HDBSCAN labels. It applies a normalized
motion-distance threshold0.15 to reference-pose16NN pairs, then requires spatial
distance <=min(0.08,2.5*min(local16NN radii)) at >=80% of sampled times.
Components smaller than128/256 remain unknown. Distances, density and sampling
still affect counts, so no-K does not mean parameter-free or semantic discovery.

`adaptive_review.py` reuses checksum-verified training evidence and the shared
GPU export/verification helpers with an explicit output directory. The new
gallery is `runs/pidg_object_centers/grouping_review/adaptive_noK_final50000`.
Old candidate PNGs are preserved; earlier galleries receive a navigation link.
Use `--prepare-only` for CPU grouping, or `--render-existing` to render an
already prepared no-K summary without refitting density models.

See [official sklearn1.8 HDBSCAN parameters](https://scikit-learn.org/1.8/modules/generated/sklearn.cluster.HDBSCAN.html).

## Motion plus predicted color, still without requested group count

```powershell
. ./reproduction/scripts/enable_cuda.ps1 -ProjectRoot (Get-Location).Path
& ./envs/pidg-cu128/python.exe -m pytest tests/test_motion_color_grouping.py tests/test_adaptive_grouping.py tests/test_motion_grouping.py tests/test_object_centers.py -q
& ./envs/pidg-cu128/python.exe -u -m variants.object_centers.color_review
```

`color_features.py` adds an optional mean-RGB feature block. The density API
defaults to color_weight=0 and the spatial graph defaults to color_radius=None,
retaining their no-color paths. Enabled density candidates concatenate
`[normalized motion features, 1.0*mean_RGB]`; the graph adds an RGB-distance
gate <=0.35 to the previous motion and persistent-spatial gates. All density,
motion, spatial and minimum-size parameters otherwise match the no-color study.
Neither method requests a group count. No PIDG model architecture/loss changes.

Color is the **existing C1 predicted** positive-SH RGB, clipped to[0,1], averaged
over the12 training view/time pairs. Every kernel/view pair is included, even
when occluded/off-screen. This is neither GT color nor an intrinsic material
albedo, and viewpoint effects or existing appearance errors can affect groups.
Similar-colored ducks remain ambiguous; distinct parts of one object can also
have different colors. No learned material segmentation is introduced.

The separate `motion_color_noK_final50000` gallery exports the same six
candidate triptychs and comparison panels, plus72 GT/C1/no-color/with-color
before/after images. Palette IDs are independent across candidates, so the
same displayed color is not a cross-candidate identity. Old outputs are read,
not rerendered or overwritten. Flags-off checks cover synthetic inputs and one
real no-color density/graph candidate against saved labels and memberships.
