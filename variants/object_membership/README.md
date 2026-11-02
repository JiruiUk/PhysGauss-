# Conditional local object membership

This non-neural diagnostic distinguishes historical object association from
current visible support and from material-point identity. It has no training
integration. `Config.enabled=False` returns before accessing inputs; no PIDG
source, model or optimizer changes.

Use a separate identity domain for each independently seeded sequence. A local
component number is never a global object ID. The caller must establish image
continuity and reciprocal links, and require complete compositing contribution,
center support and observed source/target motion before passing a label.
Three consecutive supported observations qualify a row. A missing observation
does not erase its history, but has no current visible label. A supported
different identity permanently withdraws unique ownership in that domain.
Outputs at a frame use no future evidence. These are conservative conditional
evidence rules, not calibrated probabilities or physical ground truth.

`audit.py` consumes existing native1600 C1 contribution caches. It checks the
image chain in both directions, uses positive covariance/depth/center and
per-point source/target flow support, and compares existing A2/A15 banks only
after an independent image link. It deliberately omits the historical cohort
gate and reports that difference. Do not compare its percentages to the old
cohort-gated results as a model improvement. Early and late domains stay
separate, including when row IDs overlap.

The output NPZ embeds the historical checkpoint SHA; `results.json` and
`input_lock.json` bind all arrays to source/cache/bank/data hashes. Gaussian row
IDs cannot be transferred to a different model or densified training state.
Do not feed these labels into material, shape or contact training as verified
material identity. Local object membership may join different surface patches;
Neo-Hookean F still requires valid material neighborhoods and correspondence.

Run from the repository root:

```powershell
envs/pidg-cu128/python.exe -m pytest tests/test_object_membership.py tests/test_surface_correspondence.py tests/test_motion_correspondence.py tests/test_clip_camera_geometry.py -q -p no:cacheprovider
envs/pidg-cu128/python.exe -u variants/object_membership/audit.py
envs/pidg-cu128/python.exe -u variants/object_membership/verify.py
envs/pidg-cu128/python.exe variants/object_membership/visualize.py
```

The cache audit is the inference smoke check: no neural training is applicable
to this module and no optimizer steps occur. Preserve all historical caches.
Protocol/results/assessment: `experiments/object_membership`; arrays/figures:
`output/object_membership`. The visual review covers six selected frames, not
every frame or true 3D/material identities.
