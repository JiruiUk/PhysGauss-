# Observable association training pilot

Independent repaired-C1 fork. `association_training_enabled=false` delegates
the loss path directly to C1 and creates no bank/association parameters.
The trainer fork changes only the Tuning import. This is a reconstruction and
correspondence diagnostic, not a new constitutive physics milestone.

The frozen early2–27 chain binds 2,065 of 86,143 Gaussian rows to four local
objects. Unknown rows remain unknown; late identities are not training targets.
Two complete native1600 color renders encode the four banks. Early verified
views penalize leakage and loss of visible contribution; symmetric valid raw
optical flow supplies weak temporal continuity on all119 training pairs.

T0 and T1 restore identical full50k C1 model, optimizers, RNG and camera stack.
Both perform2,000 updates through52k, without topology changes or LR resets.
Only the association flag differs. No forced permutation after occlusion.

Run from the workspace after enabling the existing CUDA environment:

```powershell
. .\reproduction\scripts\enable_cuda.ps1
& envs/pidg-cu128/python.exe -u -m variants.association_training.run
& envs/pidg-cu128/python.exe -u -m variants.association_training.evaluate
```

Preflight:32 CPU/GPU tests, actual renderer optimizer update, flags-off short
training, and40-step on-path training split by a full-state resume at50020.
See `experiments/association_training/preflight.json` and `optimizer_probe.json`.
`training_freeze.json` checks source/data hashes before and after formal training.

Full-state final checkpoints are evaluated independently of pre-update PLY
snapshots. Complete color-adjoint mass conservation, bank support and image
quality are separate metrics. A clean model permutation is only a candidate;
physical post-occlusion identity requires independent evidence.
