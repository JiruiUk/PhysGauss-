param([switch]$Resume, [ValidateSet('mechanics_cloth','balls_reaction','dry_ice','motion_kuro','rubber_duck')][string[]]$Scenes=@())
$ErrorActionPreference='Stop'
$taskProjectRoot=Split-Path -Parent $PSScriptRoot
$taskQueue=Join-Path $taskProjectRoot 'variants\upgrade'
$taskStop=Join-Path $taskQueue 'stop_queue.request'
if (Test-Path -LiteralPath $taskStop) {
    if (-not $Resume) { throw 'Queue has a saved stop request; use -Resume to acknowledge and continue.' }
    Rename-Item -LiteralPath $taskStop -NewName ("stop_queue.acknowledged_"+(Get-Date -Format yyyyMMdd_HHmmss))
}
. (Join-Path $taskProjectRoot 'reproduction\scripts\enable_cuda.ps1') -ProjectRoot $taskProjectRoot
$taskPython=Join-Path $taskProjectRoot 'envs\pidg-cu128\python.exe'
$taskRunner=Join-Path $PSScriptRoot 'run_upgrade.py'
$taskStamp=Get-Date -Format yyyyMMdd_HHmmss
$taskArguments=@('-u', ('"'+$taskRunner+'"'), '--mode', 'all', '--stages') + @(1..15 | ForEach-Object { "$_" })
if ($Scenes.Count -gt 0) { $taskArguments += @('--scenes') + $Scenes }
$taskProcess=Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskQueue "queue_$taskStamp.stdout.log") -RedirectStandardError (Join-Path $taskQueue "queue_$taskStamp.stderr.log")
Write-Output "Started upgrade controller PID $($taskProcess.Id). Scene selection uses -Scenes or the persisted benchmark scope; each selected run trains for 50k steps."
