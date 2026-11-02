# Coherent trajectory / momentum experiment

The completed L1/M1 models already derive material velocity and acceleration
from the complete rendered centers `R(q(X,t))*X+b(X,t)`. This experiment does
not claim to remove a second independent velocity network for the first time.

Optional `coherent_momentum` retains that one trajectory and the original
predicted stress/shared latent, density=1 and deviatoric velocity-rate stress.
It makes the **discrete** momentum and trajectory advancement compatible:

```
v_minus = (x(t)-x(t-h))/h
v_plus  = (x(t+h)-x(t))/h
v(t)    = (v_minus+v_plus)/2
a(t)    = (v_plus-v_minus)/h
x_physics(t+h) = 2*x(t)-x(t-h) + h^2*div(sigma(t))
L = Huber((x(t+h)-x_physics(t+h))/h^2)
```

Use normalized `h=1/199` in momentum and observed-flow velocity. Interval
centers at sequence endpoints move into `[h,1-h]` and are explicitly logged.
No time extrapolation is used. Coordinates remain raw FP32 field evaluations;
differences and normalized local solves use FP64. This does not remove errors
already present in FP32 positions or prove derivative convergence.

One 32-neighbor anchored quadratic fit in current world coordinates estimates
the predicted-stress Jacobian and velocity Hessian, replacing nested noisy
linear fits. For the existing rate term,
`div(dev(sym(grad(v)))) = 0.5*laplacian(v) + grad(div(v))/6` in 3D.
The relative ridge is `1e-5`; full polynomial rank ratio must exceed `1e-6`.
Invalid-neighborhood fractions are reported and insufficient coverage fails.

This changes the discrete operator, its neighborhood and arithmetic precision;
it is not a single scalar ablation. The M1 gradient policy remains the same.
There are no new network parameters, forces, constitutive models or material
unknowns. Turn the new flag off to call M1 exactly; all flags off retain the
original inherited PIDG path. Parent trees and historical models are untouched.

The rollout equality is a training objective, not an exact physical solver.
Displaying `h^2*residual` as rollout error does not provide independent evidence.
Both old and new operators must be applied to both models on common anchors;
success also requires retained image quality and independent observed motion.
No ground-truth stress or 3D momentum is available in this experiment.

`experiments/coherent_momentum/plan.json` records prospective criteria, budget,
known limitations and required checks before full training. One candidate starts
from the same full extended 3k checkpoint as M1 and ends at 50k. Selection uses
validation only. No additional full candidates are launched automatically.
