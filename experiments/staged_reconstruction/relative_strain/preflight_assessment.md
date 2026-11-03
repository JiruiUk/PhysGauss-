# I9 qualification before formal training

18 CPU tests passed (8 new relative-prior tests and10 inherited transport tests).
They cover affine recovery, common canonical gauge cancellation, rigid rotation,
expansion/contraction symmetry, gradients into both configurations, gradcheck,
finite backward at repeated identity singular values, deterministic sampling,
actual shared-map/canonical gradients and invalid inputs.

Native original CUDA renderer checks passed. With all features off, RGB/alpha
and loss are exact; maximum gradient difference3.73e-9. Initial enabled RGB and
alpha maximum errors6.26e-7 and8.05e-7 satisfy the unchanged original tolerances.
With I8 transport active and only the new strain flag off, I8 loss is exact
after actual updates, gradient maximum difference1.49e-8. Canonical covariance
and shared-map gradients also pass an isolated actual rasterizer-backward test.

The new prior alone gives finite/nonzero gradients into canonical coordinates
and shared motion, with none into opacity, features, scales, rotations, frozen
static geometry, original inactive fields or physics parameters. It does not
advance CPU/CUDA RNG. Render and physics centers differ by at most2.39e-7 after
20 forced physics-path smoke updates (2 physical calls). This qualification is
not an admitted formal physics result.

Independent-process runs: original flags-off control100; enabled40, resumed80,
resumed200. `preflight.json` passes original model/optimizer initialization,
exact sampler/Python/CPU/CUDA/NumPy RNG, finite Adam state and unchanged static
parameters. No formal physics calls occur in these warmup smokes. At200 the
sampled relative stretch extrema are0.935–1.094; that sample is not a full-state
geometry guarantee. Formal training starts from original step zero independently
of these smoke states. No validation/test outcomes were used to choose weights.

Receipts: `native_preflight.json`, `preflight.json`, smoke logs, `unit_tests.log`.
No completeness or appearance improvement is established by these checks.
