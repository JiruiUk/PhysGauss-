param([ValidateSet('mechanics_cloth','balls_reaction','dry_ice','motion_kuro','rubber_duck')][string]$Scene = 'motion_kuro')
$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$taskRun = Join-Path $taskProject "runs\pidg_official\$Scene"
$taskMetadata = Join-Path $taskRun 'reproduction.json'
$taskInfo = Get-Content -LiteralPath $taskMetadata -Raw | ConvertFrom-Json
if ($taskInfo.status -eq 'saved_stopped') { Write-Output 'Training has already been saved and stopped'; return }
if ($taskInfo.status -ne 'training') { throw "Scene is not training: $($taskInfo.status)" }
$taskProcesses = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*PhysGauss*' -and $_.CommandLine -like '*entrypoint.py train_pidg.py*' -and $_.CommandLine -like "*data\PIDG\$Scene *" })
if ($taskProcesses.Count -ne 1) { throw 'Expected exactly one matching training process' }
$taskCheckpointDir = Join-Path $taskRun 'checkpoints'
New-Item -ItemType Directory -Path $taskCheckpointDir -Force | Out-Null
Set-Content -LiteralPath (Join-Path $taskCheckpointDir 'save_and_stop.request') -Value 'User requested full checkpoint and GPU release' -Encoding UTF8
Write-Output 'Requested complete checkpoint and exit at the end of the current iteration'
for ($taskAttempt=0;$taskAttempt -lt 55;$taskAttempt++) {
    Start-Sleep -Seconds 1
    $taskInfo = Get-Content -LiteralPath $taskMetadata -Raw | ConvertFrom-Json
    if ($taskInfo.status -eq 'saved_stopped') {
        $taskCheckpoint = Get-Content -LiteralPath (Join-Path $taskCheckpointDir 'resume.json') -Raw | ConvertFrom-Json
        if ((Get-FileHash -LiteralPath $taskCheckpoint.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskCheckpoint.sha256) { throw 'Saved checkpoint checksum mismatch' }
        Write-Output "Saved at iteration $($taskCheckpoint.iteration); training exited and GPU allocation released"
        return
    }
    if ($taskInfo.status -eq 'failed') { throw 'Training failed before save completion; inspect train.log' }
}
throw 'Save not confirmed yet. Leave the process intact and inspect its log; do not terminate before checkpoint verification.'
