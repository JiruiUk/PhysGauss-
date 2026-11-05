# J6: user-authorized I10 dynamic + accepted J4 static joint training

The human accepts the currently displayed J4 static background and explicitly
requests incorporating the I10 model into joint training. This authorizes a new
bounded experiment despite the old J4/I10 visual non-promotion; it does not
rewrite those results or resume paused reliable_motion/S1--S6 experiments.

Use the actual I10_static_residual final2000 checkpoint for all non-static model
state, the complete dynamic/physics Adam, RNG, remaining camera stack and physics
graph caches. Use the accepted J4 joint_depth_trust final2000 checkpoint only
for the static bank and its complete static Adam. Verify identical model schema
and coordinate frame before composition. No duplicated Gaussian bank, body
count, semantic/GT mask, topology or canonical reinitialization is introduced.
The inherited color-conditioned training/selection lineage remains disclosed.

Two fixed matched arms start from this exact composite state: static frozen,
and static opacity/SH open with the unchanged J4 weight0.1 opacity-only parent
D/alpha self-reference. Each performs2000 new native1600 updates using full
0.8L1+0.2DSSIM RGB on the120 train camera/time pairs, original rates/epsilon,
and the inherited I10 RNG/sampler. Do not repeat the old I10 target replacement:
it was designed to isolate a frozen static residual and is not the objective
of this new full-image joint comparison. The RGB-only loss is fixed in advance.

Static geometry remains frozen. Original velocity/stress weights, Adam and
graph caches are retained exactly but are not updated in this RGB pilot.
This is not a claim of new physics training. Parent-render depth is only a
self-reference, not independent static geometry truth. All features default off.
No baseline/frozen-source/historical-weight edits or parameter sweep.

Before full training: test source composition/no mutation/default-off and
schema rejection; verify native initial RGB/alpha/D exact flags-off parity,
finite intended gradients and self-depth opacity-only route. Perform20 actual
smoke updates per arm, then two independent-process complete model/twoAdam/
bindings/RNG/sampler/graph restores and the user-approved fixed-native-gradient
exact next Adam replay. Preserve earlier full40 CUDA trajectory failures; no
deterministic-native-training claim or epsilon changes.

Commit preregistration and implementation/qualification separately. Freeze all
source/checkpoint/train input hashes before training. Use task-local locks,
stop requests and atomic full-state checkpoints every250 updates. New process
logs/versioned failures are retained. Do not silently resume a stopped run.

Evaluate all30 val images at native1600 with unmodified observed RGB; no test
RGB or val-based training-point selection. Report full-image metrics and
source references separately from geometry. Replay all2000 camera samples,
both Adams, immutable parameters and graph/RNG state. Produce fixed86/188
early/middle/late static/dynamic/joint RGB, alpha, D/alpha, full Gaussian PLY
and XYZ-only world diagnostics, plus two H264/yuv420p/faststart200-frame videos
with complete decoding and actual image inspection. The parent label denotes
I10+J4 composite, never the original S3-U final6k. Keep old artifacts untouched.

Numerical comparison retains max0.3dB val PSNR loss versus the matched frozen
arm. Static visual non-regression and dynamic support/completeness must be
actually reviewed; full-image PSNR, darkness, deleted support or zero temporal
static variation cannot certify geometry. No independent fixed-view cross-time
GT is available. A completed trial can remain not promoted. Do not extend or
start another candidate automatically, or treat one pair as a stable causal
estimate given the recorded I10 native numerical trajectory sensitivity.
