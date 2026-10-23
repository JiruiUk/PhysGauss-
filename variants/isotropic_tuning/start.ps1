param([switch]$Smoke,[string]$Train,[double]$Hours=2)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$taskRuns=Join-Path $taskRoot 'runs\pidg_isotropic_tuning'
New-Item -ItemType Directory -Path $taskRuns -Force | Out-Null
$taskState=Join-Path $taskRuns 'controller.json'
if (Test-Path -LiteralPath $taskState) {
    $taskPrevious=Get-Content -LiteralPath $taskState -Raw | ConvertFrom-Json
    $taskExisting=Get-CimInstance Win32_Process -Filter "ProcessId = $($taskPrevious.pid)" -ErrorAction SilentlyContinue
    if ($taskExisting -and $taskExisting.CommandLine -like '*isotropic_tuning*run.py*') { throw 'Isotropic tuning is already running.' }
}
if ($Smoke -and $Train) { throw 'Choose Smoke or Train.' }
if (-not $Smoke -and -not $Train) { throw 'Specify -Smoke or -Train I0_T3_control/I1_iso_1e4/I2_iso_1e3/I3_iso_1e2.' }
. (Join-Path $taskRoot 'reproduction\scripts\enable_cuda.ps1') -ProjectRoot $taskRoot
$taskArgs=@('-u',(Join-Path $PSScriptRoot 'run.py'),'--hours',"$Hours")
if ($Smoke) { $taskArgs+='--smoke' } else { $taskArgs+=@('--train',$Train) }
$taskStamp=Get-Date -Format yyyyMMdd_HHmmss
$taskProcess=Start-Process -FilePath (Join-Path $taskRoot 'envs\pidg-cu128\python.exe') -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRuns "queue_$taskStamp.stdout.log") -RedirectStandardError (Join-Path $taskRuns "queue_$taskStamp.stderr.log")
Write-Output "Isotropic tuning PID $($taskProcess.Id); monitor experiments/isotropic_tuning/monitor.html"
