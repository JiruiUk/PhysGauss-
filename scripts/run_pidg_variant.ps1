param(
    [ValidateSet('smoke','benchmark')][string]$Mode = 'smoke',
    [string[]]$Scenes = @('mechanics_cloth'),
    [int]$Seed = 0,
    [Parameter(Mandatory=$true)][string]$Tag
)
$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $taskProjectRoot 'reproduction\scripts\enable_cuda.ps1') -ProjectRoot $taskProjectRoot
$taskPython = Join-Path $taskProjectRoot 'envs\pidg-cu128\python.exe'
$taskRunner = Join-Path $PSScriptRoot 'run_pidg_variant.py'
& $taskPython -u $taskRunner --mode $Mode --seed $Seed --tag $Tag --scenes $Scenes
if ($LASTEXITCODE -ne 0) { throw "PIDG variant run exited with $LASTEXITCODE; inspect its logs" }
