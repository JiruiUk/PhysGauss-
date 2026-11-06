param(
    [string[]]$Scenes = @('mechanics_cloth'),
    [int]$Iterations = 50000,
    [switch]$Smoke,
    [ValidateSet('official','paper_hparams')][string]$Profile = 'official',
    [int]$Seed = 0,
    [string]$Tag = '',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $PSScriptRoot 'enable_cuda.ps1') -ProjectRoot $taskProjectRoot
$taskPython = Join-Path $taskProjectRoot 'envs\pidg-cu128\python.exe'
$taskRunner = Join-Path $PSScriptRoot 'run_baseline.py'
$taskArguments = @('-u', $taskRunner, '--scenes') + $Scenes + @('--iterations', "$Iterations", '--profile', $Profile, '--seed', "$Seed")
if ($Smoke) { $taskArguments += '--smoke' }
if ($Tag) { $taskArguments += @('--tag', $Tag) }
if ($Resume) { $taskArguments += '--resume' }
& $taskPython @taskArguments
if ($LASTEXITCODE -ne 0) { throw 'PIDG reproduction failed; inspect the run logs' }
