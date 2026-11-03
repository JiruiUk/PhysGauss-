# Proposal requiring human selection: serialization fidelity vs trajectory identity

This proposal is NOT adopted. Original J1 and J1-R1 full40 gates remain failed.
No formal or Rubber Duck training may run while this selection is pending.

J1's newly preregistered recovery test combines two properties: correctly
restoring complete training state and reproducing40 native CUDA optimization
steps across separate runs. Measurements show exact initial state/RNG/sampling,
exact immediate model/two-Adam reload and same-state RGB, but tiny differences
in the original native backward, already before any process interruption.
The fixed common-epsilon control does not satisfy the original40-step test.
This evidence does not prove restoration is broken or that reconstruction
cannot improve. It also does not prove all divergence is numerically harmless.

Option A (recommended): separately preregister a serialization-fidelity
acceptance with all of the following, before formal training:

1. Two independent processes reconstruct every model/buffer, static and dynamic
   Adam value/group, all RNG and camera stack/visits/trace exactly from one pinned
   real native step20 checkpoint; verify actual optimizer parameter binding.
2. Export one native full-RGB backward's gradients from that checkpoint and next
   fixed view into a hashed replay-only artifact. Feed identical saved gradients
   to both reloaded states and demand the resulting full model/two-Adam step
   exactly equal. This audits serialization and optimizer update semantics,
   without calling different stochastic native trajectories identical.
3. Check native same-state RGB/alpha exact and independent raw gradients within
   the original1e-6/1e-5, finite intended groups and no frozen geometry mutation.
4. Keep the original40-run failure and numerical instability visible in all
   reports. Do not call the old gate passed. Continue to save full checkpoints
   and audit6000 matching camera sequences. Any nonfinite/large same-state
   native gradient or replay mismatch blocks training.
5. Retain all original static quality/coverage/spill and added dynamic/ray/thin
   gates unchanged. Require actual independent static GT and native images;
   no improvement follows from passing serialization. Only after formal
   synthetic gates could a separate Rubber Duck protocol be considered.

Option B: retain the original full40 trajectory-identity criterion and keep
formal training stopped. Investigate a separately opt-in deterministic native
accumulation implementation, checking forward/gradient equivalence; no kernel
rewrite, broader native changes, tolerance relaxation or repeated parameter
search is authorized by this proposal. Scope/time would be larger.

The user must choose because changing the frozen recovery acceptance conflicts
with the explicit instruction not to modify experiment gates. No elapsed-time
default, implicit approval or automation may adopt Option A.
