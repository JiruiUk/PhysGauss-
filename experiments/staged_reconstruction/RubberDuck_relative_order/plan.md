# L8-R: scale-free recurring near/far relationships, one local training pilot

User explicitly accepts relative relationships and requests an actual training
trial. L8 interval-pair design had0 fit pairs and was closed without updates;
its uncertainty gates/results remain unchanged. This separately declared design
uses raw ordering, not metric calibration, affine intervals or depth targets.
No parameter sweep, relaxation/retry of L8, model replacement or old promotion.

Use all122 original T4 native-qualified persistent RGB tracks (not just L7's one
metric-depth track), and L7-J stationary-compatible observed anchor groups.
Neither set implies object IDs or static labels. Fixed official Small raw120
maps are reused. No alpha/color class, masks, object counts, GT or val selection.

For each moving-track/anchor-group pair in its12 observed frames, compare both
5x5 raw relative-inverse-depth patches. Require each patch range <=.1 of the
full-image q90-q10 and disjoint patch ranges separated by >=.1 of that range.
Larger raw inverse depth means nearer. At least4 even observations must give
the same strong sign; at least4 odd observations must independently support that
sign, with no strong opposite evidence in either half. Fit chooses sign, check
can reject it but cannot replace it. Missing/weak/edge evidence remains unknown.
Use only fit-anchor groups for training, check-anchor groups for diagnostics;
no metric-depth, calibration hull or absolute-scale claims in this mechanism.
Retain all pair hypotheses/rejections. Stable track/group order caps4 pairs per
image, fixed confidence.5. Unknown is never classified dynamic/background.

Same native own-ID visibility/NCC/reprojection/alpha guards and reversible
controller as L8. Fixed default-off one-sided robust log ranking weight.001,
margin log1.02, scale.05; no absolute target or large depth-gap fitting.
Original Gaussian representation/dynamic pool/PIDG renderer unchanged.
Both arms start from complete J6/twoAdam/RNG/camera/physics states. Static and
physics weights/Adam, graph caches/topology frozen. PureRGB control vs RGB+
relative order each600 native1600 updates, ordinal render400; original sampling,
Adam epsilon1e-15, no automatic extension or other temporal/physical losses.

First source/pair/native input freeze, tests, native flags-off/RGB-alpha-D and
finite intended gradients,20-step smoke and user's accepted independent exact
state/fixed-native-gradient Adam recovery. Qualification-only forced violation
tests D gradients without changing any training target or model weight.

One native-supported pair suffices only for this explicitly local exploratory
trial; original T4/L7 global gates remain failed. Evaluate all30 native1600 val,
both .3dB nonregression gates, fixed ordinal fit/check denominators and missing
support, all122 original tracks, static equality,86/188 early/middle/late native
RGB/alpha/D, complete PLY and two200-frame decoded H264 comparisons. Actual
visual completeness/cleanup must pass before promotion; ranking/RGB cannot prove
geometry, thickness, identity or whole-duck relationships. Historical color
lineage disclosed; no new physics, semantic count or GT. Failures preserved,
each milestone separately committed, stop saves full state and releases GPU.
