param([int]$Port=8767)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskLogs=Join-Path $taskRoot 'runs\pidg_kinematic_tuning\monitor'
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
$taskUrl="http://127.0.0.1:$Port/"
$taskExisting=$null
try{$taskExisting=Invoke-RestMethod ($taskUrl+'api/status') -TimeoutSec 2}catch{}
if($taskExisting.monitor_kind -eq 'pidg_kinematic_live'){Write-Output "Monitor already running: $taskUrl";return}
if($taskExisting){throw "Port $Port serves another application."}
$taskStamp=Get-Date -Format yyyyMMdd_HHmmss
$taskProcess=Start-Process -FilePath (Join-Path $taskRoot 'envs\pidg-cu128\python.exe') -ArgumentList @('-u',(Join-Path $PSScriptRoot 'monitor_kinematic.py'),'--port',"$Port") -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskLogs "live_$taskStamp.stdout.log") -RedirectStandardError (Join-Path $taskLogs "live_$taskStamp.stderr.log")
for($taskAttempt=0;$taskAttempt -lt 30;$taskAttempt++){
    if($taskProcess.HasExited){throw "Monitor exited; inspect $taskLogs."}
    try{$taskReady=Invoke-RestMethod ($taskUrl+'api/status') -TimeoutSec 1
        if($taskReady.monitor_kind -eq 'pidg_kinematic_live'){Write-Output "Monitor $taskUrl PID $($taskProcess.Id), refresh every 5 seconds.";return}
    }catch{}
    Start-Sleep -Milliseconds 300
}
throw 'Monitor startup timed out.'
