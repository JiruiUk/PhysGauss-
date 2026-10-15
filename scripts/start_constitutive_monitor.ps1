param([int]$Port=8765,[switch]$OpenBrowser)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskMonitor=Join-Path $taskRoot 'runs\pidg_constitutive\monitor'
New-Item -ItemType Directory -Path $taskMonitor -Force | Out-Null
$taskUrl="http://127.0.0.1:$Port/"
$taskExisting=$null
try { $taskExisting=Invoke-RestMethod ($taskUrl+'api/status') -TimeoutSec 2 } catch {}
if ($taskExisting.milestones) { Write-Output "Constitutive monitor already running: $taskUrl"; if($OpenBrowser){Start-Process $taskUrl}; return }
if ($taskExisting) { throw "Port $Port is serving a different monitor; choose another -Port." }
$taskStamp=Get-Date -Format yyyyMMdd_HHmmss
$taskProcess=Start-Process -FilePath (Join-Path $taskRoot 'envs\pidg-cu128\python.exe') -ArgumentList @('-u',('"'+(Join-Path $PSScriptRoot 'monitor_constitutive.py')+'"'),'--port',"$Port") -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskMonitor "server_$taskStamp.stdout.log") -RedirectStandardError (Join-Path $taskMonitor "server_$taskStamp.stderr.log")
for ($taskAttempt=0;$taskAttempt -lt 30;$taskAttempt++) {
    if ($taskProcess.HasExited) { throw "Monitor exited; see $taskMonitor\server_$taskStamp.stderr.log" }
    try { $taskReady=Invoke-RestMethod ($taskUrl+'api/status') -TimeoutSec 1; if($taskReady.milestones){Write-Output "Monitor $taskUrl PID $($taskProcess.Id), every 5 seconds";if($OpenBrowser){Start-Process $taskUrl};return} } catch {}
    Start-Sleep -Milliseconds 300
}
throw 'Monitor startup timed out; inspect monitor logs.'
