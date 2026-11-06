# Numerical repair, fixed before its output

Original preregistered sparse trust-region implementation reached400 evaluations
in all3 fits. Gradient optimality was0.0108/0.0116/0.0133. This is a numerical
failure, not evidence that joint scale is inaccurate. Original audit.json,
source and measure_attempt1.log remain unchanged and byte-hashed.

Independent solver_v2 replaces approximate sparse LSMR trust-region subproblems
with exact dense subproblems and Jacobian column scaling. Same120x2 unknowns,
same observations, relative Huber data residuals, per-frame count weights,
quadratic second differences, ridge, bounds, neutral initialization,400 function
evaluations,1e-10 solver tolerances and all calibration/motion/coverage gates.
This is not a new candidate, no threshold/weight search or changed data decision.
Report objective and optimality, compare with independent analytic gradient/KKT.
Do not promote nonconverged numerical output or overwrite failed audit.
