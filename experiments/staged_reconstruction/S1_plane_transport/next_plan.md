# After D7: inspect static revocation in a jointly rendered scene

D7 improves synthetic depth locally, but both numerical and Gaussian evidence
gates fail. Do not repeat, relax checks or use its depths/normals as qualified
training supervision. No new Gaussian/real18k-to24k continuation follows.

The user's latest direction is to let the dynamic branch explain moving content
while revoking static smears. S3-U freezes every static parameter; original S6
only enables static SH, not opacity/geometry. This is a representation/update
limitation, not proof that unfreezing will improve reconstruction.

Existing independent static_dynamic_coupling audit9a9c8fb6 already measured
initialization dependencies, three conditional-mask gradient probes and three
native compositing probes per legacy model. Reuse this evidence; do not rerun
or duplicate it. Large dynamic gaps remain in dynamic-only renders, so current
static occlusion is not a sufficient explanation. Frozen initialization and
earlier training influence remain distinct and unresolved.

Next single stage is S1-J0 full-RGB static-revocation feasibility audit, explicitly
preregistered before new code/results. It only enables static opacity/appearance
autograd in an in-memory copy of the exact S3-U final6k; geometry stays frozen.
No optimizer step, training loss integration, graft of newer static bank, new
physics or instance discovery. Original unchanged scene values must render
identically with gradient flags off/on; parameter/checkpoint hashes and caller
RNG state remain unchanged. Test native finite gradients and source restoration
in independent processes before the six fixed training-frame measurements.

Use full RGB0.8 L1+0.2 DSSIM without color/static/dynamic masks, evaluator GT,
inferred depth, or object number. At fixed frames, measure static-opacity/SH,
dynamic-canonical and deformation gradients and native static-only, dynamic-only
and depth-sorted combined images. Full-RGB gradients can still prefer harmful
geometry or static removal; signs/norms are not correct assignments or proof of
cleanup. Unknown/occluded and reflections must not be called dynamic/background.
The inherited S3-U weights DID use historical conditional color masks: report
this provenance, never claim a new label-free training result from this audit.

The latest unknown-static S1-C18k cannot simply be spliced into an old dynamic
checkpoint and called a matched joint experiment. Qualifying a new consistent
joint parent/geometry and a separate two-condition Gaussian protocol, complete
native gradients/flags-off/full Adam-RNG-camera recovery and independent static
GT/structure evidence would still be necessary before any real continuation.
Do not infer qualification from J0 gradients or failed D7 depths. All original
gates, frozen source/weights and paused work are preserved.

Any subsequent learned joint candidate must test static-only cleanup as well
as the combined image, protect supported thin surfaces, and reject apparent
improvement caused by dynamic occlusion of static smears. Joint learning is a
new hypothesis, not a rebranding of the failed per-image nuisance experiment.
