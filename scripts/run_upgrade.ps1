param([int[]]$Stages = @(1), [ValidateSet('smoke','long','all')][string]$Mode='all', [ValidateSet('mechanics_cloth','balls_reaction','dry_ice','motion_kuro','rubber_duck')][string[]]$Scenes=@())
$ErrorActionPreference='Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $taskProjectRoot 'reproduction\scripts\enable_cuda.ps1') -ProjectRoot $taskProjectRoot
$taskPython=Join-Path $taskProjectRoot 'envs\pidg-cu128\python.exe'
$taskSceneArguments=@()
if ($Scenes.Count -gt 0) { $taskSceneArguments=@('--scenes') + $Scenes }
& $taskPython -u (Join-Path $PSScriptRoot 'run_upgrade.py') --stages $Stages --mode $Mode @taskSceneArguments
if ($LASTEXITCODE -ne 0) { throw "Upgrade queue exited with $LASTEXITCODE" }
