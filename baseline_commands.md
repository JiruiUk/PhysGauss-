# M0 baseline commands and reuse

User instructed reuse of existing completed baselines; no new training was launched for M0.

Commands executed for this audit:

```powershell
.\.venv\Scripts\python.exe scripts/audit_pidg_m0.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Reproduction recipes

These are reproducible train/render/evaluation recipes from the recorded protocol and existing launcher, not a claim that the historical shell transcript was captured. Use a new output tag; never overwrite completed runs.

```powershell
. .\reproduction\scripts\enable_cuda.ps1 -ProjectRoot (Get-Location).Path
```

### mechanics_cloth

```powershell
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 train_pidg.py -s "F:\A_Edinburgh\PhysGauss/data/PIDG/mechanics_cloth" -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_mechanics_cloth" --conf "F:\A_Edinburgh\PhysGauss\third_party\PIDG/arguments/pidg/mechanics_cloth.py" --load2gpu_on_the_fly --resolution 1600 --iterations 50000 --save_image_dir "F:\A_Edinburgh\PhysGauss/runs/new_m0_mechanics_cloth/diagnostics" --checkpoint_interval 1000
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 render.py -s "F:\A_Edinburgh\PhysGauss/data/PIDG/mechanics_cloth" -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_mechanics_cloth" --conf "F:\A_Edinburgh\PhysGauss\third_party\PIDG/arguments/pidg/mechanics_cloth.py" --load2gpu_on_the_fly --resolution 1600 --iteration best --skip_train
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 metrics.py -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_mechanics_cloth"
```

### balls_reaction

```powershell
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 train_pidg.py -s "F:\A_Edinburgh\PhysGauss/data/PIDG/balls_reaction" -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_balls_reaction" --conf "F:\A_Edinburgh\PhysGauss\third_party\PIDG/arguments/pidg/balls_reaction.py" --load2gpu_on_the_fly --resolution 1600 --iterations 50000 --save_image_dir "F:\A_Edinburgh\PhysGauss/runs/new_m0_balls_reaction/diagnostics" --checkpoint_interval 1000
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 render.py -s "F:\A_Edinburgh\PhysGauss/data/PIDG/balls_reaction" -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_balls_reaction" --conf "F:\A_Edinburgh\PhysGauss\third_party\PIDG/arguments/pidg/balls_reaction.py" --load2gpu_on_the_fly --resolution 1600 --iteration best --skip_train
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 metrics.py -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_balls_reaction"
```

### dry_ice

```powershell
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 train_pidg.py -s "F:\A_Edinburgh\PhysGauss/data/PIDG/dry_ice" -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_dry_ice" --conf "F:\A_Edinburgh\PhysGauss\third_party\PIDG/arguments/pidg/dry_ice.py" --load2gpu_on_the_fly --resolution 1600 --iterations 50000 --save_image_dir "F:\A_Edinburgh\PhysGauss/runs/new_m0_dry_ice/diagnostics" --checkpoint_interval 1000
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 render.py -s "F:\A_Edinburgh\PhysGauss/data/PIDG/dry_ice" -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_dry_ice" --conf "F:\A_Edinburgh\PhysGauss\third_party\PIDG/arguments/pidg/dry_ice.py" --load2gpu_on_the_fly --resolution 1600 --iteration best --skip_train
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 metrics.py -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_dry_ice"
```

### motion_kuro

```powershell
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 train_pidg.py -s "F:\A_Edinburgh\PhysGauss/data/PIDG/motion_kuro" -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_motion_kuro" --conf "F:\A_Edinburgh\PhysGauss\third_party\PIDG/arguments/pidg/motion_kuro.py" --load2gpu_on_the_fly --resolution 1600 --iterations 50000 --save_image_dir "F:\A_Edinburgh\PhysGauss/runs/new_m0_motion_kuro/diagnostics" --checkpoint_interval 1000
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 render.py -s "F:\A_Edinburgh\PhysGauss/data/PIDG/motion_kuro" -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_motion_kuro" --conf "F:\A_Edinburgh\PhysGauss\third_party\PIDG/arguments/pidg/motion_kuro.py" --load2gpu_on_the_fly --resolution 1600 --iteration best --skip_train
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 metrics.py -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_motion_kuro"
```

### rubber_duck

```powershell
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 train_pidg.py -s "F:\A_Edinburgh\PhysGauss/data/PIDG/rubber_duck" -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_rubber_duck" --conf "F:\A_Edinburgh\PhysGauss\third_party\PIDG/arguments/pidg/rubber_dack.py" --load2gpu_on_the_fly --resolution 1600 --iterations 50000 --save_image_dir "F:\A_Edinburgh\PhysGauss/runs/new_m0_rubber_duck/diagnostics" --checkpoint_interval 1000
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 render.py -s "F:\A_Edinburgh\PhysGauss/data/PIDG/rubber_duck" -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_rubber_duck" --conf "F:\A_Edinburgh\PhysGauss\third_party\PIDG/arguments/pidg/rubber_dack.py" --load2gpu_on_the_fly --resolution 1600 --iteration best --skip_train
envs/pidg-cu128/python.exe -u scripts/pidg_variant_entrypoint.py --repo third_party/PIDG --seed 0 metrics.py -m "F:\A_Edinburgh\PhysGauss/runs/new_m0_rubber_duck"
```

## Observed completed 50k baselines

| Scene | PSNR | SSIM | LPIPS | MS-SSIM |
|---|---:|---:|---:|---:|
| mechanics_cloth | 39.478764 | 0.985344 | 0.037297 | 0.990284 |
| balls_reaction | 32.910084 | 0.979940 | 0.029670 | 0.986920 |
| dry_ice | 26.160622 | 0.903799 | 0.137407 | 0.927060 |
| motion_kuro | 26.912134 | 0.951768 | 0.057407 | 0.944780 |
| rubber_duck | 27.256781 | 0.976551 | 0.029624 | 0.970483 |

The original released code has existing compatibility patches; these are disclosed in PIDG_REPO_MAP.md. Historical results lack runtime hashes; current manifests are not retrospective provenance. New integration smoke runs use short budgets and must not be compared as 50k gains.
