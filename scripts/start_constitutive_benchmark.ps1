param([ValidateSet('smoke','long')][string]$Mode='long',[switch]$Resume)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskRuns=Join-Path $taskRoot 'runs\pidg_constitutive'
New-Item -ItemType Directory -Path $taskRuns -Force | Out-Null
$taskController=Join-Path $taskRuns 'controller.json'
if (Test-Path -LiteralPath $taskController) {
    $taskPrevious=Get-Content -LiteralPath $taskController -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskExisting=Get-CimInstance Win32_Process -Filter "ProcessId = $($taskPrevious.pid)" -ErrorAction SilentlyContinue
    if ($taskExisting -and $taskExisting.CommandLine -like '*run_constitutive_matrix.py*') {
        throw "Matrix worker PID $($taskPrevious.pid) is still running. Wait for its exit before starting or resuming."
    }
}
$taskStop=Join-Path $taskRuns 'stop_queue.request'
if (Test-Path -LiteralPath $taskStop) {
    if (-not $Resume) { throw 'A saved stop is pending. Use -Resume to continue.' }
    Rename-Item -LiteralPath $taskStop -NewName ('stop_queue.acknowledged_'+(Get-Date -Format yyyyMMdd_HHmmss))
}
. (Join-Path $taskRoot 'reproduction\scripts\enable_cuda.ps1') -ProjectRoot $taskRoot
$taskArgs=@('-u',('"'+(Join-Path $PSScriptRoot 'run_constitutive_matrix.py')+'"'),'--mode',$Mode)
if ($Resume) { $taskArgs+='--resume' }
$taskStamp=Get-Date -Format yyyyMMdd_HHmmss
$taskProcess=Start-Process -FilePath (Join-Path $taskRoot 'envs\pidg-cu128\python.exe') -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRuns "queue_$taskStamp.stdout.log") -RedirectStandardError (Join-Path $taskRuns "queue_$taskStamp.stderr.log")
for ($taskAttempt=0; $taskAttempt -lt 30; $taskAttempt++) {
    Start-Sleep -Milliseconds 500
    $taskProcess.Refresh()
    if ($taskProcess.HasExited) {
        if ($taskProcess.ExitCode -ne 0) { throw "Matrix worker exited with code $($taskProcess.ExitCode). See queue_$taskStamp.stderr.log." }
        break
    }
    if (Test-Path -LiteralPath $taskController) {
        $taskReady=Get-Content -LiteralPath $taskController -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($taskReady.pid -eq $taskProcess.Id) { break }
    }
}
if (-not $taskProcess.HasExited -and $taskReady.pid -ne $taskProcess.Id) { throw "Worker readiness not confirmed; inspect PID $($taskProcess.Id) and queue logs before starting another worker." }
Write-Output "Constitutive $Mode matrix worker PID $($taskProcess.Id). Scope is stored in experiments/pidg_experiment_matrix.yaml."
