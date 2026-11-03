# I7: covariance support is inconsistent, but normalization alone is insufficient

This is a read-only follow-up to the position-Jacobian diagnosis, using the same
failed frozen16k checkpoint and original renderer. No new model was trained.

The released CUDA `computeCov3D` uses `q = rot` with quaternion normalization
commented out. S3-U supplies normalized canonical quaternion plus an unconstrained
rotation-head increment. Its resulting polynomial matrix need not be orthogonal;
this permits extra scale/distortion beyond the explicitly bounded scale head.
Do not change historical PIDG behavior to hide this property.

At the nine fixed times the quaternion-norm99th percentile ranges2.45–3.68.
Comparing actual covariance against the same scales with a unit quaternion,
26%–46% of rows have at least one generalized axis ratio below0.25 or above4.
Actual ellipsoid longest/shortest-axis99th percentiles range14.7–50.4.
For nonreference times,22%–85% of rows differ that strongly from transporting
frame2 covariance through the measured relative position differential.
The opacity>0.1 stratum also shows substantial disagreement. Opacity is not
visibility; these are point statistics, not rendered-contribution fractions.
Neither covariance transport nor frame2 is independently established material GT.

Three mathematical covariance tests and five derivative tests pass. Python
precomputed covariances were checked against native1600 rendering at t43/t146:
maximum projected-conic differences1.07e-6/2.75e-5, RGB mean absolute
differences2.86e-9/1.61e-8. The original maximum-pixel tolerance3e-4 **failed**
at t146: maximum RGB8.11e-4 and alpha3.92e-3. Both original failed logs are
preserved; the final receipt reports that failure rather than loosening its
tolerance. Do not claim exact alternate-renderer parity or admit it for training.
The diagnostic formula follows the released raw-quaternion CUDA convention;
it is not a replacement renderer.

An actual inference-only intervention then normalized dynamic quaternions while
keeping centers, explicit scales, opacity, appearance and original CUDA path.
Probes were c188/t43, c188/t146 and c86/t86, using existing metadata poses and
no GT RGB. The contact sheet and native c188/t43/t146 images were inspected.
Some elongated/puffy supports shrink, but many detached points and large missing
regions remain; t146 does not become a complete object. No completeness success
or causal attribution to this single issue is supported.

Artifacts: `output/staged_reconstruction/invertible_motion/covariance_intervention/`.
Raw distributions, source hashes and native numerical differences are in
`covariance_diagnostic.json`; image review is in `covariance_intervention_review.json`.

Next hypothesis must address persistent reference-surface support and its
transport with the motion field, rather than only increasing strain penalties
or postprocessing ellipsoid size. A new covariance-transport representation
would require its own opt-in implementation, gradient and native renderer
checks, controlled initialization/resume, and original16k admission. Its ability
to establish a complete canonical shape remains unproven. Trusted depth would
add useful geometry evidence; the existing failed video-depth targets are not
eligible supervision. No new formal training or physics updates were launched.
