param([string]$Experiment='A1',[string]$Tag,[int]$Iterations=0,[switch]$Long,[switch]$Resume)
$ErrorActionPreference='Stop'
$taskProjectRoot=Split-Path -Parent $PSScriptRoot
if (-not $Tag) { $Tag=$Experiment+'_'+(Get-Date -Format yyyyMMdd_HHmmss) }
. (Join-Path $taskProjectRoot 'reproduction\scripts\enable_cuda.ps1') -ProjectRoot $taskProjectRoot
$taskArguments=@('--experiment',$Experiment,'--tag',$Tag)
if ($Iterations -gt 0) { $taskArguments+=@('--iterations',"$Iterations") }
if ($Long) { $taskArguments+='--long' }
if ($Resume) { $taskArguments+='--resume' }
& (Join-Path $taskProjectRoot 'envs\pidg-cu128\python.exe') -u (Join-Path $PSScriptRoot 'run_constitutive_experiment.py') @taskArguments
if ($LASTEXITCODE -ne 0) { throw "Constitutive integration run exited with $LASTEXITCODE" }
