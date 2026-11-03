$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'F:\A_Edinburgh\PhysGauss'
. ./reproduction/scripts/enable_cuda.ps1 -ProjectRoot 'F:\A_Edinburgh\PhysGauss'
$env:PYTHONPATH = 'F:\A_Edinburgh\PhysGauss;F:\A_Edinburgh\PhysGauss\src'
& ./envs/pidg-cu128/python.exe -u experiments/staged_reconstruction/S1_clean_visibility/run_pipeline.py --enable-clean-static
exit $LASTEXITCODE
