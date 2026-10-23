param([int]$Port=8766)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskReports=Join-Path $taskRoot 'experiments\isotropic_tuning'
$taskLogs=Join-Path $taskRoot 'runs\pidg_isotropic_tuning\monitor'
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
$taskUrl="http://127.0.0.1:$Port/"
$taskExisting=$null
try { $taskExisting=Invoke-RestMethod ($taskUrl+'api/status') -TimeoutSec 2 } catch {}
if ($taskExisting.monitor_kind -eq 'pidg_isotropic_charts') { Write-Output "Charts already running: $taskUrl"; return }
if ($taskExisting) { throw "Port $Port serves another application; choose another Port." }
$taskStamp=Get-Date -Format yyyyMMdd_HHmmss
$taskProcess=Start-Process -FilePath (Join-Path $taskRoot 'envs\pidg-cu128\python.exe') -ArgumentList @('-u',(Join-Path $PSScriptRoot 'monitor_isotropic.py'),'--port',"$Port") -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogs "charts_$taskStamp.stdout.log") -RedirectStandardError (Join-Path $taskLogs "charts_$taskStamp.stderr.log")
for ($taskAttempt=0;$taskAttempt -lt 40;$taskAttempt++) {
    if ($taskProcess.HasExited) { throw "Chart server exited; inspect $taskLogs." }
    try {
        $taskReady=Invoke-RestMethod ($taskUrl+'api/status') -TimeoutSec 1
        if($taskReady.monitor_kind -eq 'pidg_isotropic_charts') { Write-Output "Charts $taskUrl PID $($taskProcess.Id), refresh every 5 seconds; local page $taskReports\charts.html"; return }
    } catch {}
    Start-Sleep -Milliseconds 300
}
throw 'Chart monitor startup timed out; inspect monitor logs.'
