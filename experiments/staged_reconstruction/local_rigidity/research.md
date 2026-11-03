# I5: rotation-aware local rigidity from original S3-U step zero

Dynamic3DGaussians,3DV2024 (<https://dynamic3dgaussians.github.io/paper.pdf>)
couples relative neighbor displacements to local rotations, adds rotational
similarity and long-term isometry, and keeps appearance/size persistent.
Its experiments use27 synchronized training cameras and initial depth samples.
Our200-time sequence has only one image at each time; its guarantees do not
transfer. This is an adaptation to PIDG, not a reproduction of that system.

I1 only penalized excess edge-length changes around frozen parent medians.
I2 shared sparse transforms but did not enforce agreement of relative neighbor
vectors. I3/I4 ordinal depth could improve ordering without preventing ribbons;
I4 also failed original static preservation at16k. CoTracker3 strict image
evidence is too sparse and has no middle-band tracks, so no tracking objective
is admitted. These failures motivate testing geometric motion coherence.

Use original S3-U full step-zero state, identical16k warmup/6k physics budget,
camera/width schedule, optimizer and frozen static model. Build directed20-NN
within radius.08 on initial canonical coordinates; no body count, components,
material segmentation, learned stiffness or force. Neighborhoods are proposals,
not known material associations. Report node coverage before training.

For an edge i→j use the current learnable canonical relative vector Xj−Xi.
At sampled training time t, require xj(t)−xi(t)≈Ri(t)(Xj−Xi), where Ri is the
original PIDG position-warp quaternion. Frozen initial lengths normalize the
robust residual; they are not enforced as canonical rest geometry. Add squared
matrix disagreement between neighboring warp rotations. This long-range
canonical comparison differs from the paper's adjacent-time online objective
and does not use its covariance-rotation field. No quaternion sign ambiguity.

Predeclared weights: rigid.3, rotation.03,400-update linear ramp,4096 directed
edges per update on a deterministic cycle. No prior-dependent RNG calls,
trainable weights or opacity escape. Gradients target original grid/network
and canonical coordinates. Two flags independently disable the two terms;
the experiment entry point is opt-in and defaults to control. No depth/tracks.

Unit tests precede full integration, then flags-off native loss/gradient
comparison, forced20-step physics-path integration smoke, independent40→80→200
resume and original sample/Adam/RNG audit. At16k apply unchanged original
warmup gates and actual native image review. Only an admitted model proceeds
to6k physics and the unchanged six-camera screen/full1200-frame acceptance.
No test RGB is used for iterative selection. A more rigid yet incomplete
model remains a failure; rendering plausibility does not establish true3D.
