# I2 shared-motion differential audit

This is a read-only diagnosis of the rejected I2 checkpoint, not a new
training objective or a constitutive deformation-gradient module. It reads
no RGB and performs zero optimizer updates. The checkpoint and controls are
bound by SHA256 in `blend_diagnostic.json`.

For the nine preselected times, the 10th percentile of the smallest singular
value of the weighted rotation matrix is 0.9965–0.9993. No sampled Gaussian
has that value below 0.25. Thus severe rotation-blending contraction is not
supported as the dominant failure mechanism in this checkpoint. A switch to
dual-quaternion blending alone is not justified by these measurements.

Including the spatial derivative of the normalized RBF weights changes the
result substantially: 4.75%–31.50% of sampled Gaussian locations have a
smallest local singular value below 0.25, and 1.16%–16.24% have negative
local determinant. The translation/weight-gradient part of the complete
mapping therefore matters. These are model-conditioned local measurements,
not independent evidence of true material motion, and the derivative is
piecewise away from KNN support changes. They do not establish the cause of
every observed missing region or prove that preventing folds restores a
complete object.

The two unit tests pass: constant rigid transformations reduce to their
rotation, and the analytic complete derivative matches autograd for varying
control transforms. The initial test run exposed an indexing error; its log
is preserved separately from the corrected successful run.

I6 remains a separate frozen experiment. This diagnostic does not alter its
source, controls, optimizer, training data, acceptance gates or checkpoint.
