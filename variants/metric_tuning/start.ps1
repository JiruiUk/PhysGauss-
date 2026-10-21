param([double]$Hours=24,[switch]$Resume,[switch]$PreflightOnly)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$taskRuns=Join-Path $taskRoot 'runs\pidg_metric_tuning'
New-Item -ItemType Directory -Path $taskRuns -Force | Out-Null
$taskState=Join-Path $taskRuns 'controller.json'
if (Test-Path -LiteralPath $taskState) {
    $taskPrevious=Get-Content -LiteralPath $taskState -Raw | ConvertFrom-Json
    $taskExisting=Get-CimInstance Win32_Process -Filter "ProcessId = $($taskPrevious.pid)" -ErrorAction SilentlyContinue
    if ($taskExisting -and $taskExisting.CommandLine -like '*metric_tuning*run.py*') { throw 'Metric tuning is already running.' }
}
. (Join-Path $taskRoot 'reproduction\scripts\enable_cuda.ps1') -ProjectRoot $taskRoot
if ($Resume -and (Test-Path -LiteralPath (Join-Path $taskRuns 'stop_queue.request'))) {
    Move-Item -LiteralPath (Join-Path $taskRuns 'stop_queue.request') -Destination (Join-Path $taskRuns ('stop_queue.acknowledged_'+(Get-Date -Format yyyyMMdd_HHmmss)+'.request'))
}
$taskArgs=@('-u',(Join-Path $PSScriptRoot 'run.py'),'--hours',"$Hours")
if ($Resume) { $taskArgs+='--resume' }
if ($PreflightOnly) { $taskArgs+='--preflight-only' }
$taskStamp=Get-Date -Format yyyyMMdd_HHmmss
$taskProcess=Start-Process -FilePath (Join-Path $taskRoot 'envs\pidg-cu128\python.exe') -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRuns "queue_$taskStamp.stdout.log") -RedirectStandardError (Join-Path $taskRuns "queue_$taskStamp.stderr.log")
if (-not $PreflightOnly) {
    $taskExport=Start-Process -FilePath (Join-Path $taskRoot 'envs\pidg-cu128\python.exe') -ArgumentList @('-u',(Join-Path $PSScriptRoot 'export_comparisons.py'),'--wait','--controller-pid',"$($taskProcess.Id)") -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRuns "export_$taskStamp.stdout.log") -RedirectStandardError (Join-Path $taskRuns "export_$taskStamp.stderr.log")
    Write-Output "Final GT / metric-winner gallery exporter PID $($taskExport.Id)."
}
Write-Output "Metric tuning PID $($taskProcess.Id); budget $Hours hours; monitor experiments/metric_tuning/monitor.html"
