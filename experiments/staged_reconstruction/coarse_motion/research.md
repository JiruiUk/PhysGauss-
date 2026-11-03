# I6: small shared motion capacity from original step zero

I5's local rotation/geometry objective reduced some spreading but still left
large missing regions, degraded validation dynamic PSNR, and failed the static
preservation gate. Do not add more local penalties or relax that gate.

SC-GS (<https://arxiv.org/html/2312.14937v2>) motivates shared control transforms;
Neural Parametric Gaussians (<https://arxiv.org/html/2312.01196v2>) motivates
strong coarse geometry/motion sharing before fine Gaussian detail. NPG also
reports flat-template failure, so neither principle guarantees completeness.
We test a PIDG capacity adaptation, not either full paper system.

The I2 adapter queried256 controls on the already-trained S3-U shape, with
four-neighbor RBF blending and an extra I1 graph objective. I6 queries16
controls from original step zero. It retains original canonical Gaussians,
appearance, scaling bounds, PIDG network/grid/velocity/stress and image/physics
objectives, without the I1/I5/depth/track losses. It uses the same original
full initial state, camera/width schedule and16k+6k budget. Original S3-U is the
matched control; I2 is not a single-switch comparison due to initialization,
control capacity, bandwidth and additional-loss differences.

Controls are deterministic canonical FPS samples, with no components, body
count inference, instance labels or learned material segmentation.16 is a
chosen motion capacity, not a physical object count. The pilot geometry audit
found that the original fourth-neighbor bandwidth gives one isolated control
radius.71 scene units. Avoid such extreme averaging by using a uniform fixed
bandwidth equal to half the initial90th-percentile nearest-control distance
(.093419). This is fixed from initial coordinates before training, not fitted
to validation/test performance. Each query blends its four nearest controls.

Use the frozen I2 adapter's full transformed-center interpolation and its
quaternion sign alignment. Start fully sparse at update1; zeroed original
heads initialize an identity mapping up to floating point. No new learnable
parameters. The exact same adapter is used by rendering and original physics
queries; independent native checks must verify this after optimizer updates.
Disabled installation returns without reading assets or changing the scene.

Tests precede full training integration. Required preflight: original disabled
RGB/alpha/loss/gradient parity, enabled render/physics mapping and gradients,
20 forced physics-path updates,100 disabled updates and independently resumed
40→80→200 fully sparse updates with full Adam/RNG/camera audit. At16k require
original numerical gates plus native review before the original6k physics.
Final screen/full1200 acceptance and test-selection policy are unchanged.
