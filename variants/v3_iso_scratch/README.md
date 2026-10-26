# V3 / V3+iso: genuine from-scratch 50k comparison

This replaces the paused continuation study at the user's request. No trained
parent model is loaded. Each arm receives **50k total training iterations**, not
50k additional refinement iterations.

Both arms use the baseline W256_D0 architecture, V3 image losses (L1, SSIM and
0.02 VGG-LPIPS), shared frame-index rendering/physics time, and no kinematic
consistency term. The original PIDG initial-training optimizer recipe restores
warm-up3000, densification through15000, and original initial learning rates.
These settings are common to both arms. They differ from the prior two-stage
V3 recipe; compare the two fresh arms to isolate the isotropic loss.

| Arm | Total iterations | Iso weight | First iso computation |
|---|---:|---:|---:|
| V3_scratch_control | 50000 | 0 | never |
| V3_scratch_iso_1e3 | 50000 | 0.001 | 1 |

The unchanged shape prior from `variants/isotropic_tuning` is applied to rendered
axis magnitudes `abs(exp(log_scale)+d_scaling)`, normalized per Gaussian by its
own mean axis length. During canonical warm-up, `d_scaling=0`, so it acts on
canonical scales; once deformation starts, it includes deformation scale offsets.
No extra network parameters or solid-material losses are introduced.

## Identical initialization before any training

The initialization job creates new network, Gaussian, velocity and stress state
with seed0, using the existing independently generated uniform random point cloud
in the derived dataset. That file contains no learned geometry. `parent_run` is
empty and `warm_start=False` throughout this experiment.

The optional capture hook runs **before the training loop**, after no-grad
validation at iteration0. It saves a complete step-zero checkpoint, then exits
without executing any gradient update. We check that iteration is exactly zero
and both optimizer state dictionaries are empty. Both training runs copy these
same tensor/optimizer/RNG/camera bytes; the complete checkpoint SHA256 is
identical. Only the iso weight in the configuration sidecar and the recorded
output-directory relocation differ. Reading this newly created step-zero file
is initialization sharing, not loading a previously trained model.

The iso run starts first; the control follows. Their training trajectories may
still differ due to CUDA nondeterminism after the common starting state.

## Data, selection and output

The existing 120 train / 30 validation / 50 official test split and common
`(frame_id-1)/199` clock remain fixed. The uniform acquisition-cadence assumption
is not independently verified. The training directory contains no final-test
RGB. Four best checkpoints are selected only on validation, every1000 steps;
final50k is preserved independently. After both arms finish, selections are
locked, then all five declared snapshots are tested and exported as GT/render
pairs. Historical test exposure remains a limitation.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File variants/v3_iso_scratch/launch.ps1 -Smoke -Hours 24
powershell -NoProfile -ExecutionPolicy Bypass -File variants/v3_iso_scratch/launch.ps1 -Hours 24
```

- Plan, zero-step evidence, preflight and results: `experiments/v3_iso_scratch`
- Current iter and 5-second validation charts: `experiments/v3_iso_scratch/monitor.html`
- New trained weights: `runs/pidg_v3_iso_scratch`
- Final pairs: `runs/pidg_v3_iso_scratch/render_comparisons`

The previous continuation is saved at iteration10489 in
`runs/pidg_v3_isotropic/V3_continue_control/rubber_duck/checkpoints/resume.pt`;
its stop request remains present. Those weights are not used here.
