# L8: explicitly authorized local ordinal-depth training pilot

User: 能用于确定鸭子和其他的相对关系也行，也可以试着训练一下看看效果。
This authorizes one limited exploratory training pair despite L7-J's failed
global coverage. It does not reverse L7/T4 outcomes or promote a new model.
No absolute pseudo-depth loss. J6 remains the retained reference until review.

One fixed candidate: preserve only robust near/far order between an originally
L7-J-admitted moving RGB track and observed stationary-compatible RGB anchors.
These are point hypotheses, not named ducks, object IDs or static labels.
Fixed Small priors, RGB/camera/time and inherited J6/T4 evidence only; historical
color lineage disclosed, no new color selectors, masks, counts, depth/normal GT.

Build pairs before native training: only L7-J passed motion tracks, their local
valid even-fit observations, current-frame fit-only calibration anchors. Both
points must remain in the current fit hull, finite, away from prior discontinuity;
minimum8px separation, normalized raw gap>=.1, and the full fit plus both
disjoint-fit calibrations must give nonoverlapping log-depth intervals separated by log1.02.
At most4 pairs/frame, stable anchor group/ID order, no model-error selection.
All proposals/rejections retained. Build an odd/check-anchor diagnostic bank
with identical rules separately; never use it to choose targets or parameters.

Both arms start from the exact full J6 model, two Adam states/bindings, RNG,
remaining120-view camera stack and physics graph cache. Dynamic pool only is
updated. Static model/Adam, physics model/Adam and topology are frozen. Use the
original full-image native1600 0.8L1+.2DSSIM; candidate adds a default-off robust
one-sided log-depth ranking hinge (weight.001, margin log1.02, scale.05), using
native400 full-scene D/alpha at observed UVs. No absolute depths or desired gaps
beyond the small fixed order margin are fitted. D/alpha remains a rendered proxy.

Own-row native compositing share>=.01, tracked-UV patch NCC>=.6 with both std>=2,
row reprojection<=4px, positive depth and both endpoint alpha>.5 qualify evidence.
Native eligibility refreshes at initial and every100 completed steps in BOTH
arms, with exact RNG/flags restoration; unknown suspends immediately, two known
contradictions revoke, two passes restore. Per-step current positive/alpha and
row reprojection guards suspend the loss. Fixed pair denominator prevents missing
targets from increasing weight. No deletion, growth, reassociation or identities.

Run only one matched600-step RGB vs RGB+ordinal pair, no extension/sweep. Initial
native support requires>=1 usable pair only for this explicitly local pilot;
all old global gates remain failed. First tests/default-off/native RGB-alpha-D
and gradients,20-step smoke per arm, independent complete-state recovery and
same-fixed-native-gradient Adam exact update (the user's accepted protocol).
Native D-gradient qualification may force a scalar violation in a marked test
fixture without changing weights/targets or using it in training.

Evaluation is fixed before training: all30 validation native1600 (test withheld),
max.3dB regression vs matched RGB and J6, all original122 tracks with fit/odd
reprojection, fixed pair ranking and missing-support denominators, current alpha
and dynamic dark-support proxies, original86/188 times2/100/200 raw RGB/alpha/D,
complete Gaussian PLYs and two200-frame H264 comparison videos fully decoded.
Actually inspect the fixed images/decompositions. Ranking/PSNR alone cannot
establish object relationships, thickness, completeness or geometric cleaning.
No promotion without original visual completeness/cleanup/static nonregression.

Every milestone separately committed. Freeze source/input/bank before smoke or
formal updates, retain all failures. On stop save complete state and release GPU;
never automatically resume paused work or change historical PIDG code/weights.
