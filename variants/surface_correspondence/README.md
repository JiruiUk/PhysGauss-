# Visible surface correspondence bank

This optional non-neural module links local visible image fragments to frozen
canonical Gaussian rows. `Config.enabled` defaults to false; `optional_bank`
returns before reading state when disabled. Historical PIDG sources and weights
are unchanged. A bank is a conditional visible surface patch, not a physical
object/material label or a trained reconstruction.

Three consecutive, independently qualified visible observations select each
bank once. Both centers and complete compositing footprints must agree with the
image fragment; projected center motion must also match raw observed flow.
Future frames test the same fixed canonical rows. Terminated image seeds and
failed surface support remain unknown, including after occlusion. Cross-anchor
row overlap is a hypothesis and never merges physical identities.
Per-point output also requires visible footprint/center support in BOTH the
source and target observation of the same local bank; valid background flow
at an unsupported source position cannot certify surface motion.

From the repository directory, using the existing environment:

```powershell
envs/pidg-cu128/python.exe -m pytest tests/test_surface_correspondence.py -q -p no:cacheprovider
. ./reproduction/scripts/enable_cuda.ps1
envs/pidg-cu128/python.exe output/surface_correspondence/contracts.py
envs/pidg-cu128/python.exe output/surface_correspondence/audit.py
envs/pidg-cu128/python.exe output/surface_correspondence/finalize.py
envs/pidg-cu128/python.exe output/surface_correspondence/bind_banks.py
envs/pidg-cu128/python.exe output/surface_correspondence/verify.py
```

The audit is read-only with respect to historical models, but regenerates its
own diagnostic caches/results. Record manual review in the corresponding
receipt after inspecting regenerated figures. The independent verifier needs
the seed caches under `runs/pidg_surface_correspondence/cache`, whose hashes are
recorded in `results.json`; it recomputes seed membership and all120 forward
gates without running neural optimization.

Row indices are bound by `bank_bindings.json` to the exact historical C1 full
checkpoint and data. They are not stable IDs across training/densification or
other models; never reuse them by point count or nearest canonical coordinates.

Inputs/protocol/assessment: `experiments/surface_correspondence`.
Bank row IDs/canonical coordinates, full trajectories, per-frame unknown labels,
figures and gallery: `output/surface_correspondence`.
