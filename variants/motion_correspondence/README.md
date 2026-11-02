# Motion correspondence diagnostic

This optional non-neural module tests motion-history candidate association before
changing PIDG shape training. It does not train/relabel Gaussians, estimate material
IDs, introduce external force, or change frozen historical training paths.

`Config.use_history` enables position forecasting; `Config.uncertainty_guard`
also inflates missing-observation uncertainty and preserves unresolved identity.
`optional_visible_propagation(enabled=False)` calls the existing visible-only
propagator exactly. Merged image components are not split by nearest positions.
There is no independent identity-evidence input in this prototype; a latched
identity cannot be cleared merely because later motion is smooth.

From the repository directory, use the existing `envs/pidg-cu128/python.exe`:

```powershell
envs/pidg-cu128/python.exe -m pytest tests/test_motion_correspondence.py -q
envs/pidg-cu128/python.exe -m variants.motion_correspondence.run
envs/pidg-cu128/python.exe output/motion_correspondence/rolling_test.py
. ./reproduction/scripts/enable_cuda.ps1
envs/pidg-cu128/python.exe output/motion_correspondence/render_support.py
envs/pidg-cu128/python.exe output/motion_correspondence/finalize.py
envs/pidg-cu128/python.exe output/motion_correspondence/verify_results.py
```

These commands regenerate diagnostic receipts, not historical model checkpoints.
Manual review receipts describe the actual inspected artifacts and must be
updated if regenerated results change. Protocols/assessment/initial failed probe
are in `experiments/motion_correspondence`; gallery and masks are in
`output/motion_correspondence`. No full reconstruction training is authorized by
passing a position prediction score: persistent surface support and reliable
identity evidence remain separate prerequisites.
