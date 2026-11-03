# S3 engineering preparation; formal training still gated

This stage retains original PIDG `VelocityNet` time-evolving planes, velocity
head and predicted stress head independently per object. Index embeddings are
disabled, and the original deviatoric rate coefficient/density are fixed at1.
No constitutive law, learned material/force or contact module is introduced.

Velocity is a material field `v(X,t)`; its time derivative is taken at fixed
material coordinates without adding convection again. Stress divergence uses
the existing tested FP64 current-world quadratic WLS operator. Every neighbor
graph is built inside one canonical object; static/other-object points cannot
enter it. Invalid rank is a failure, not a zero residual. Kinematic consistency
ties the independent velocity to the rendered center trajectory, with the
existing explicit shared-gradient cap. Raw terms, valid fractions, head values,
velocity disagreement and guard magnitude remain visible in diagnostics.

Time uses actual normalized frame timestamps `(frame-1)/199`, including image
gaps. Acquisition seconds are unavailable; neither video display FPS nor the
training-frame count is a physical clock. Density/stress/velocity are therefore
dimensionless diagnostics; they cannot establish real material parameters.

Manufactured operator tests and a GPU engineering probe on the S2 recovery80
smoke state pass, including field/deformation/grid/canonical gradients, an actual
Adam field update and exact static freeze. That probe is not formal S3 training
and does not establish S2 admission. The6000-step native continuation entry
requires a reviewed S2 `admission.json` with the exact accepted parent SHA.
It preserves the parent image optimizer/RNG/sampler, adds only the original PIDG
fields and saves every new optimizer/RNG/model state. It requires a separate
40→80 S3 recovery smoke before formal training. No S3 run is launched on a failed
S2 model, and no later stage is marked complete by these engineering tests.
