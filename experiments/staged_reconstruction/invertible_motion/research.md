# Conditional invertible position-map prototype

Status: I6 completed16k warmup and failed both the original static-preservation
gate and actual new-view completeness. This is the next independent opt-in
representation experiment; no formal result is claimed here.

The read-only I2 differential audit found little contraction from weighted
rotation matrices, but appreciable contraction and negative determinants in
the full map including normalized RBF weight derivatives. Hard KNN support
changes are a further piecewise limitation. Merely changing quaternion
averaging does not directly remove those effects.

[CaDeX, CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Lei_CaDeX_Learning_Canonical_Deformation_Coordinate_Space_for_Dynamic_Surface_Representation_CVPR_2022_paper.pdf)
constructs conditional invertible canonical maps using affine coupling.
[MoDGS](https://arxiv.org/html/2406.00434v4), Sec.3.1, uses a shared,
time-conditioned invertible MLP as its deformation map, alongside separate
3D-aware initialization and depth supervision. We propose only a PIDG
position-map adaptation, not a reproduction of either complete method.

The tested prototype composes six smooth affine couplings, cycling the
updated axis through x,y,z twice. Each conditioner observes only the other
two coordinates and time, preserving an analytic inverse. Bounded log scales
(+-0.2 per block) give a positive full determinant in [exp(-1.2),exp(1.2)]
in exact arithmetic. Translation bounds are0.2 per block. Width64,
two spatial and four temporal Fourier bands are capacity choices, unrelated
to object count. Zero output heads initialize an identity map. No learned
instances, material segmentation, depth, tracks or constitutive loss.

These properties do not guarantee a lower bound on every singular value,
complete canonical shape, correct material identity, visibility, or a
physically correct scene. Canonical positions can still optimize to a poor
shape; opacity and Gaussian support can still cause missing regions. A
coupling experiment would therefore retain all numerical and actual native
image gates, and cannot pass on analytic invertibility alone.

Before integration: six prototype tests pass for identity initialization,
three-time round trips and analytic/autograd determinants, coordinate/time/
parameter gradients, and shape validation. Native flags-off parity,
render/physics agreement, original gradient policy, optimizer/RNG audit,
resume tests, source freeze and separate commits remain required before any
formal training.

Integration retains all original PIDG parameters and covariance heads. The
original position head is intentionally inactive when enabled. A new shared
position module has33,420 parameters in its own Adam group; it is also included
in the existing deformation-gradient balancing group through `scene.networks`.
This uses the original gradient policy, not an unguarded auxiliary physics
path. Existing initial tensors/Adam groups and sampling are preserved; this
is not a claim that the enlarged initial state equals the unextended state.

Eight CPU tests now include disabled installation and optimizer/CPU-RNG
preservation. An initial GPU smoke audit found CUDA RNG was reset by
`torch.manual_seed` inside a CPU-only RNG fork. The failed run and logs are
archived. Initialization now seeds only the CPU default generator, and native
preflight explicitly asserts both CPU and CUDA RNG preservation. Repeat
native and independent40→80→200 recovery before accepting this correction.

The corrected native preflight and complete rerun passed. Disabled RGB/alpha/
loss are exact, gradient difference1.86e-9; enabled initial RGB/alpha are also
exact.20 forced physics-path optimizer updates include2 physical calls.
Render/physics center differences after updates are at most2.39e-7; inverse
round-trip errors are below1.8e-7. The100-update disabled control and independent
40→80→200 enabled resumes pass original tensors, Adam counts, sampler,
CPU/CUDA/NumPy/Python RNG and static-parameter checks. These are preflight
checks, not actual16k warmup or successful completeness.
