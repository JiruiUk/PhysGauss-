$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $taskRoot 'reproduction\scripts\enable_cuda.ps1') -ProjectRoot $taskRoot
& (Join-Path $taskRoot 'envs\pidg-cu128\python.exe') -u (Join-Path $PSScriptRoot 'evaluate_splits.py') @args
if ($LASTEXITCODE -ne 0) { throw 'Split evaluation failed; inspect retained_evaluation/render.log and split_metrics.json.' }
