# S3-U surface trajectory research: R0/R1

This optional module executes the observation and fixed-support prerequisites of
`experiments/staged_reconstruction/S3U_SURFACE_TRAJECTORY_RESEARCH_PLAN_20261103.md`.
It does not rewrite PIDG, estimate body count, learn segmentation, add a physics
module, or perform any optimizer updates. All entries require an explicit flag.

The current evidence gate **fails**. T0/T1 training has not started. See
`experiments/staged_reconstruction/surface_trajectory/assessment.md` for results.

## Fresh-output reproduction

Use the existing `envs/pidg-cu128/python.exe`. First run the tests listed in the
assessment. Existing outputs are intentionally preserved; commands refuse to
overwrite completed results. Do not remove historical receipts to rerun them.

```powershell
.\envs\pidg-cu128\python.exe -m variants.staged_reconstruction.surface_trajectory.prepare --enable-surface-trajectory
.\envs\pidg-cu128\python.exe -m variants.staged_reconstruction.surface_trajectory.prepare --enable-surface-trajectory --observation-version 2
.\envs\pidg-cu128\python.exe -m variants.staged_reconstruction.surface_trajectory.prepare --enable-surface-trajectory --observation-version 3

. .\reproduction\scripts\enable_cuda.ps1 -ProjectRoot 'F:\A_Edinburgh\PhysGauss'
.\envs\pidg-cu128\python.exe -m variants.staged_reconstruction.surface_trajectory.bind --enable-surface-trajectory
.\envs\pidg-cu128\python.exe -m variants.staged_reconstruction.surface_trajectory.bind --enable-surface-trajectory --observation-version 3
.\envs\pidg-cu128\python.exe -m variants.staged_reconstruction.surface_trajectory.review --enable-surface-trajectory
.\envs\pidg-cu128\python.exe -m variants.staged_reconstruction.surface_trajectory.review --enable-surface-trajectory --observation-version 3
.\envs\pidg-cu128\python.exe -m variants.staged_reconstruction.surface_trajectory.diagnostics --enable-surface-trajectory
```

Gallery generation does not certify image review. A reviewer must inspect the
specified native-image crops, record actual findings and scope, and judge stable
surface feature evidence separately from independent material ground truth.
Then run the independent verification and summarize scripts:

```powershell
.\envs\pidg-cu128\python.exe experiments/staged_reconstruction/verify_surface_trajectory.py
.\envs\pidg-cu128\python.exe experiments/staged_reconstruction/summarize_surface_trajectory.py
```

## What is measured

- v1: 15 source windows, longest surviving fragments only; 22 qualified tracks.
- v2: more source times and valid prefixes of 8–16 observations; 183 tracks,
  71 bootstrap bindings. Visual review found specular-feature risk. Minimum
  length, FB/NCC/closure thresholds were not relaxed.
- v3: orange/yellow appearance envelope with small internal holes filled;
  detected bright highlight neighborhoods excluded. 66 tracks, 23 bindings,
  12 duplicate/support groups. This is a changed observation policy, not a
  controlled training ablation. It does not detect all specular features.
- Supports/weights use only three bootstrap observations. Future projection
  error never decides which bindings are admitted. Visible alpha contributions
  use exact forward color-coded renders with conservation checks, not truncated
  per-pixel metadata or assumed backward transmittance reconstruction.
- Near-duplicate observations and highly shared weighted supports stay in the
  same split group. Original validation/test RGB never supplies track training
  data. Track-loss holdout is distinct from image/geometry ground truth.

Large `bootstrap_evidence.json` caches are retained locally, excluded from Git,
and hashed in the artifact inventory. They can be regenerated from the parent
checkpoint and recorded data. Small protocols, tracks, bindings, split lists,
review records and numerical receipts are versioned.

R2 GPU gradient/optimizer smoke and full-state training recovery remain pending
because R1 failed. The completed GPU forward inference checks perform zero
optimizer steps and verify unchanged model tensors and protected sources. Do
not describe them as completed training smoke tests or motion recovery.
