param([ValidateSet('all','smoke','long')][string]$Mode='all',[switch]$Resume)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$taskRuns=Join-Path $taskRoot 'runs\pidg_mlp_capacity'
New-Item -ItemType Directory -Path $taskRuns -Force | Out-Null
$taskState=Join-Path $taskRuns 'controller.json'
if (Test-Path -LiteralPath $taskState) {
    $taskPrevious=Get-Content -LiteralPath $taskState -Raw | ConvertFrom-Json
    $taskExisting=Get-CimInstance Win32_Process -Filter "ProcessId = $($taskPrevious.pid)" -ErrorAction SilentlyContinue
    if ($taskExisting -and $taskExisting.CommandLine -like '*mlp_capacity*run.py*') { throw 'Capacity queue is already running.' }
}
. (Join-Path $taskRoot 'reproduction\scripts\enable_cuda.ps1') -ProjectRoot $taskRoot
$taskArgs=@('-u',(Join-Path $PSScriptRoot 'run.py'),'--mode',$Mode)
if ($Resume) { $taskArgs+='--resume' }
$taskStamp=Get-Date -Format yyyyMMdd_HHmmss
$taskProcess=Start-Process -FilePath (Join-Path $taskRoot 'envs\pidg-cu128\python.exe') -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRuns "queue_$taskStamp.stdout.log") -RedirectStandardError (Join-Path $taskRuns "queue_$taskStamp.stderr.log")
Write-Output "Baseline MLP capacity queue PID $($taskProcess.Id). Logs: $taskRuns"
