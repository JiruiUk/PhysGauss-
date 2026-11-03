# I3 ordinal depth continuation: depth ordering improves, completeness fails

The formal run completed2,000 updates,200 inherited physics calls and1,901
full-weight auxiliary updates. Initial model/Adam state, camera/RNG replay,
frozen static parameters and source checks passed. Native known geometry
returned camera-z5.0 exactly; depth derivative was1.000000119 vs numerical
finite difference0.999927461. No original PIDG source was edited.
Checkpoint SHA256: `f6665f72cd0d6b752f9fb1fd3d26438169a3e11ff68f8b13a4b83c4de3b4167c`.

| Native1600 validation30 | S3-U | Matched I1 | I3 |
|---|---:|---:|---:|
| Full PSNR |28.1663|28.5727|27.9415|
| Dynamic PSNR |25.1812|26.5556|25.1853|
| Dynamic alpha |.988421|.990472|.987456|
| Fixed training-pair ordinal loss,120 frames |.938820|.947473|.447694|

The fixed-pair diagnostic uses the same batch and denominator across all
models; unlike changing per-update samples, this is a comparable loss measure.
It is agreement with learned pseudo-depth, not independently measured depth
accuracy. Six-anchor full-bank wrong-order fractions also decrease, but this
does not establish shape or identity correctness. The independent diagnostic
found no widespread zero-gradient tanh saturation on these anchors; a new
ranking-loss weight sweep is not the justified next step.

Actual visual review covered nine rows each of fixed86/fixed188 comparison
sheets plus the native1600 I3 fixed188 t62 render. Fixed86 t146/172/200 still
has long sheets and detached support. Fixed188 t43/62/86 shows severe missing
volumes and stretched fragments; some early regions also stretch more than
the parent. I3 is therefore rejected despite meeting numerical guardrails.
The other four camera sheets were generated but not manually reviewed for
this rejection. No full6×200 acceptance scan or test-RGB evaluation was run.

Additional read-only scale alignment against frozen static geometry produced
held-pixel median errors .09–.22 scene units and P90 .44–.92 on six training
frames, with ~20% dynamic extrapolation in two late frames. This is too weak
to promote that calibration to metric-depth truth. No model was trained on
those calibrated target depths; the failure and qualification are preserved.

Next hypothesis: introducing the same ordinal geometry signal during the
original warmup may avoid the inherited geometry local minimum. This needs a
fresh-initialization experiment with the same initialization, original phases,
training schedule and controls, not another tweak to the failed final model.
The current result is an adaptation of MoDGS, not a full MoDGS reproduction.
