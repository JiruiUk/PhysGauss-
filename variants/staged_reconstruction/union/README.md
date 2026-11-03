# S3-U: unified dynamic Gaussian population

User-selected scope: the scene contains an unspecified number of dynamic bodies;
use one shared canonical population without estimating that number. This is not
an instance-discovery or material-segmentation model. No original S2 identities,
per-instance initializations or trained dynamic weights are loaded.

Only the accepted S1 static bank is shared and frozen. Initial dynamic pixels
come from the same conditional yellow-appearance union as the existing image
objective. A global budget of15,507 points matches the earlier total budget;
pixel sampling never computes connected components or allocates per-body quotas.
Continuous depth is an explicit uncertain static-neighborhood prior, not GT.
One unchanged original PIDG deformation network and one velocity/stress field
serve the full population. One larger hash grid has a comparable aggregate cap;
actual counts are recorded in each run's initial.json. These architectural and
initialization differences prevent a single-switch causal comparison to S3.

Run16k image-only warmup (800px through12k, then1600px), review its native val and
fixed-camera output, then run6k S3 with the original0.01 kinematic/1e-5 CMR weights,
every10 steps and the existing0.1 shared-gradient cap. Warmup must meet unchanged
20dB dynamic/.85 dynamic-alpha/1dB static-drop gates, plus separate actual image
review. No test RGB enters initialization, gradients or warmup admission.

Current-world quadratic derivative stencils contain dynamic rows only. Without
instances, they cannot exclude other physical bodies at contact or close approach.
Rank failure still fails the experiment, rather than reporting a zero residual.
Stress/velocity consistency is internal and sequence-normalized; it does not
certify geometry, force, contact or body identities. No contact, constitutive,
unknown force, track loss or densification is enabled.

All commands require explicit --enable-union-stage. Preparation freezes source,
input and protocol hashes. Each phase requires independent-process40→80 recovery
verification before formal training. The root training.lock prevents parallel
warmup/physics trainers. stop.request requests an atomic complete-state stop;
its presence prevents automatic restart. Existing paused S1–S6 work is untouched.

```powershell
. .\reproduction\scripts\enable_cuda.ps1
envs/pidg-cu128/python.exe -m variants.staged_reconstruction.union.prepare --enable-union-stage
envs/pidg-cu128/python.exe experiments/staged_reconstruction/verify_union.py --native-preflight
envs/pidg-cu128/python.exe -m variants.staged_reconstruction.union.train --enable-union-stage --phase warmup --smoke --stop-after 40
```

Then capture40 with verify_union.py, resume to80 and verify. The formal command
omits --smoke and --stop-after. Physics additionally requires a reviewed
warmup_admission.json. Review/export entry:
experiments/staged_reconstruction/review_union.py.
