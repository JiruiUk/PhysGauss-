# Background residual continuation: preflight passed

Eight CPU reference/gradient tests passed. At native1600, the disabled wrapper
returned the exact original loss and identical RGB/alpha; gradient maximum
absolute difference was5.9604645e-8 at maximum reference magnitude0.3234283,
within the existing CUDA atomic tolerance. No original PIDG source was edited.

The enabled expression retained foreground/empty/unknown reference values
exactly and detached the static render. An actual original-physics backward
at absolute6010 yielded finite nonzero canonical/grid/deformation/physics
gradient norms0.07425/0.009816/0.43433/0.003350. Frozen static parameters
received no gradients. This native check performed zero optimizer updates.

Both arms actually completed100 optimizer updates at width1600, including ten
original-physics updates. Candidate ran in three independent processes, saving
and restarting at40 and80. The audit checked exact parent model tensors, Adam
state and groups, initial camera stack and all RNG; replayed the sampler; checked
final RNG and Adam counters, finite model/state and unchanged frozen static
tensors. Each restart checkpoint hash matches its recorded launch. Full logs,
source freeze and preflight receipts remain separate from formal training.

This demonstrates implementation/state correctness, not improved geometry.
Formal budget remains2000 new updates per arm. Actual GT validation, original
fixed6x9 screening and subsequent6x200 visual acceptance remain unproven.
