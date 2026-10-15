param([string]$Experiment='A1',[string]$Tag,[int]$Iterations=300)
$ErrorActionPreference='Stop'
$taskProjectRoot=Split-Path -Parent $PSScriptRoot
if (-not $Tag) { $Tag=$Experiment+'_'+(Get-Date -Format yyyyMMdd_HHmmss) }
. (Join-Path $taskProjectRoot 'reproduction\scripts\enable_cuda.ps1') -ProjectRoot $taskProjectRoot
& (Join-Path $taskProjectRoot 'envs\pidg-cu128\python.exe') -u (Join-Path $PSScriptRoot 'run_constitutive.py') --experiment $Experiment --tag $Tag --iterations $Iterations
if ($LASTEXITCODE -ne 0) { throw "Constitutive integration run exited with $LASTEXITCODE" }
