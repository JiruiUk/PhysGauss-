# I9: pointwise relative differential prior

Primary source read: [Nerfies, ICCV2021, Sec3.3 Eq4-6](https://arxiv.org/html/2011.12948v3#S3.SS3).
Its continuous warp uses log Jacobian singular values to penalize local changes
of shape, with symmetric expansion/contraction cost and robust remapping.
It also documents monocular depth/scale ambiguity and does not guarantee complete
unobserved surfaces. Our application is an adaptation, not a Nerfies replication.

## Evidence and single change

I8's final exact differential audit, unlike I7's, measures substantial local
distortion. Common support transport alone still produces sheets/holes and fails
the original static-preservation gate. Test one new prior on the SAME I8
architecture from the SAME original S3-U step-zero state. Do not change static
initialization, static loss, position capacity or covariance transport together.

For each sampled canonical center, use R_t = J_t J_ref^-1, ref=frame2. This
removes a common canonical-coordinate gauge and penalizes deformation between
model configurations. Frame2 is NOT known to be relaxed or stress-free. Both
Jacobians and the canonical coordinates keep gradients. The log singular value
energy E is remapped as 2E/(E+4c^2), c=0.3, weight0.02. Scale0.3 is a declared
adaptation (paper uses0.03), fixed before qualification or validation outcomes.
The robust cost can saturate and may still permit extreme distortion; inspect
actual singular values and images. This is no material/constitutive stress law.

At each update, take512 centers with deterministic cycling indices, regardless
of opacity/visibility. Query observed training time plus a deterministic auxiliary
time cycling all200 positions. Equal sample weights prevent opacity suppression
from directly turning off the prior. No graph joins uncertain object identities;
the spatial map remains one shared field. No object count, topology, tracking,
depth pseudo-GT, stiffness, force or new parameter group is added. A pointwise
prior can still bias genuine local deformation and cannot invent missing shape.

## Controls, gates and falsification

The I8 final16k is the same-architecture/budget no-prior ablation; S3-U final6k
remains the fixed user-selected reference. Disabled I9 must also reproduce I8
when its transport is enabled, as well as original PIDG when all flags are off.
Retain exact16k original warmup/sampling/resolution and conditional6k physics.
Unit tests first; then native parity/gradients, isolated prior gradients, forced
physics smoke,100-update original control and independent40/80/200 prior resume.
Audit full initialization/Adam/RNG/static state and freeze before formal training.

Original numerical warmup gates and actual native review control physics
admission. Preserve failed receipts. Final acceptance still requires val dynamic
PSNR drop<=0.3dB vs S3-U, alpha>=0.95, actual native6x9 screen and then6x200
RGB/alpha inspection without missing regions, fragments, sheets or merging.
Use train/val only for decisions. No test RGB or synchronized unseen-view GT.
Lower strain or higher PSNR alone does not establish completeness.
