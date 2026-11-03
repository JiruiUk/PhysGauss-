# I8: local-affine Gaussian support transport

Evidence: I7's failed16k position map has no sampled minimum principal stretch
below0.25 at the fixed15,507 points/nine times, while its independent covariance
heads yield large support distortions. Inference-only normalization reduces
some elongated supports but leaves the object visibly incomplete. No complete
canonical shape or material identity has been established.

[PhysGaussian, CVPR2024, Sec.3.4 Eq.6–8](https://arxiv.org/html/2311.12198v3)
transports Gaussian support by the local differential of its deformation map.
We test that representation relationship in PIDG, without its MPM simulator,
material laws, internal filling or assumption of an already complete static
reconstruction. This is not a reproduction of the full paper or proof of
physical mass conservation. Opacity is rendering opacity, not density.

Use exactly the I7 shared invertible position architecture and initialization,
with one new representation flag. Set x(t)=phi(X,t), J=D_X phi(X,t), and
Sigma(t)=J Sigma_canonical J^T. Sigma_canonical comes from the same learned
canonical log-scales passed through S3-U's existing bounded-scale function at
zero warp, and its normalized canonical quaternion. Disable the independent
time-dependent scale/rotation outputs as well as the original position head
when enabled. Original tensors and Adam groups remain present; their inactive
gradients/state must be audited, not asserted to be updated. New shared
position parameters and its existing optimizer group match I7.

The spatial derivative is an exact analytic chain through the six archived
coupling layers, preserving its coordinate/time/parameter derivatives. This
avoids expensive reverse-mode Jacobian graphs over all Gaussian centers and
repeated-eigenvalue decompositions. Precomputed covariance uses the existing
renderer API. A local wrapper dispatches marked dynamic/composed banks through
that path, including evaluation's dynamic-only alpha; unmarked static and all
flags-off calls retain the original renderer path. No source patches.

No new depth, tracking, graph loss, stiffness, force, segmentation or material
branch. Same original S3-U step-zero tensors/Adam/RNG/sampler, fixed frozen S1,
12k800+4k1600 warmup and conditional6k physics, image/physics objectives and
shared-gradient guard. Source archives and failed historical gates remain.

Prototype tests precede native integration: exact derivatives versus autograd
at three times; identity, affine shear and rigid-rotation covariance transport;
full coordinate/time/scale/quaternion and position-network gradient checks;
time-shape rejection. Native checks must include unchanged flags-off RGB,
alpha/loss/gradients, initial-enabled covariance input errors below the fixed
3e-4 maximum and1e-6 mean tolerances, covariance-only gradients into its intended
variables, standalone dynamic-alpha dispatch,20 forced physics-path updates,
center agreement, and inactive original grids/networks. Do not loosen these
tolerances because the historical I7 t146 covariance diagnostic failed them.

Then100-update disabled control, independent40→80→200 enabled recovery and full
state/sampler/Adam/RNG audit. Only after passing, freeze and separately commit
before formal training. Original16k admission and actual native image review
precede physics. Final6k candidate must satisfy the unchanged six-camera/nine-
time screen,6x200 scan, actual image completeness and val dynamic PSNR/alpha
gates. Test RGB remains excluded during iteration. Correct transport cannot
invent missing canonical geometry; a smoother incomplete model still fails.

Preflight outcome:10 CPU tests pass. Actual native disabled RGB/alpha/loss are
exact; gradient difference is1.40e-9 on the repeated native check. Enabled
initial maximum RGB/alpha errors are6.26e-7/8.05e-7, within the predeclared
tolerances. Covariance-only gradients were checked through the actual CUDA
renderer with center/opacity/features detached, in addition to matrix-only
tests; canonical coordinates, scales, rotations and the shared map all receive
finite nonzero group gradients.20 forced physics-path updates include2 physics
calls and preserve inactive original grid/network gradients. Render/physics
centers agree within2.39e-7.100 disabled updates and independent enabled
40→80→200 recovery pass full initial-tensor/Adam/RNG/sampler/static audits.
An earlier passing native check is retained as attempt1; the second additionally
tests the isolated native covariance backward path. No failed gate was relaxed.
These checks admit formal optimization, not geometry success.
