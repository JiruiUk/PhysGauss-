# Fixed-view completeness metrics

Independent read-only metrics. No PIDG/renderer/trainer or physics code changes.
`evaluate(..., enabled=False)` is inert; the audit CLI is also disabled by default.
This module is not a new constitutive milestone and does not train a new model.

Two sets of quantities remain explicitly separate:

- At an actual camera/time observation, independent RGBA alpha supplies missing
  foreground, low-opacity core, background spill and silhouette IoU targets.
- At other camera/time combinations, small-island mass, thin-support mass,
  enclosed holes and boundary complexity are descriptive shape proxies. Mean,
  p95, maximum and worst frame are reported for each fixed camera separately.

No full-object completeness score is invented. Empty renders have undefined
proxy ratios. Real gaps/holes, detached legitimate objects, frame exit and
occlusion can affect the proxies. Renders are never repaired and no filled mask
is used as a pseudo target. Reducing these proxies alone can favor a smooth
opaque blob, so they must not automatically replace image quality or train a
novel-view silhouette loss.

```powershell
envs/pidg-cu128/python.exe -m pytest tests/test_view_completeness.py -q -p no:cacheprovider
envs/pidg-cu128/python.exe -m variants.view_completeness.audit --enabled
```

The audit consumes frozen1600-rendered alpha maps after their existing400x225
8-bit BOX downsampling. Fixed thresholds and report protocol are recorded in
`experiments/view_completeness/plan.json`. Validation30 times are reported
separately; all200 diagnostic observations are historically exposed data, not a
new untouched benchmark. All consumed renders/GT and original checkpoints are
hashed before/after. Existing input/output is not modified.

Existing observed-alpha training is already differentiable in
`variants/integrity_tuning/model.py::coverage_objective`. A future training
experiment must keep its observed view/time targets, image-quality guard and
independent new feature flags. Novel-view losses need justified geometry/object
priors or independent observations; not equality of different-view masks or
equality of masks at different times. This diagnostic introduces no such loss.
