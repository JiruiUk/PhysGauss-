param([int]$Port=8765, [switch]$OpenBrowser)
$ErrorActionPreference='Stop'
$taskProjectRoot=Split-Path -Parent $PSScriptRoot
$taskMonitorDirectory=Join-Path $taskProjectRoot 'variants\upgrade\monitor'
New-Item -ItemType Directory -Path $taskMonitorDirectory -Force | Out-Null
$taskUrl="http://127.0.0.1:$Port/"
$taskExisting=$null
try { $taskExisting=Invoke-RestMethod -Uri ($taskUrl+'api/status') -TimeoutSec 2 } catch {}
if ($taskExisting.monitor_pid) {
    Write-Output "Monitor already running: $taskUrl (PID $($taskExisting.monitor_pid))"
} else {
    $taskPython=Join-Path $taskProjectRoot 'envs\pidg-cu128\python.exe'
    $taskMonitorScript=Join-Path $PSScriptRoot 'monitor_upgrade.py'
    $taskStamp=Get-Date -Format yyyyMMdd_HHmmss
    $taskProcess=Start-Process -FilePath $taskPython -ArgumentList @('-u', ('"'+$taskMonitorScript+'"'), '--port', "$Port", '--interval', '5') -WorkingDirectory $taskProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskMonitorDirectory "server_$taskStamp.stdout.log") -RedirectStandardError (Join-Path $taskMonitorDirectory "server_$taskStamp.stderr.log")
    $taskReady=$false
    for ($taskAttempt=0; $taskAttempt -lt 30; $taskAttempt++) {
        if ($taskProcess.HasExited) { throw "Monitor exited. See $taskMonitorDirectory\server_$taskStamp.stderr.log" }
        try {
            $taskStatus=Invoke-RestMethod -Uri ($taskUrl+'api/status') -TimeoutSec 1
            if ($taskStatus.monitor_pid -eq $taskProcess.Id) { $taskReady=$true; break }
        } catch {}
        Start-Sleep -Milliseconds 300
    }
    if (-not $taskReady) { throw "Monitor did not become ready; inspect logs in $taskMonitorDirectory" }
    Write-Output "Monitor started: $taskUrl (PID $($taskProcess.Id)); refresh every 5 seconds."
}
if ($OpenBrowser) { Start-Process $taskUrl }
