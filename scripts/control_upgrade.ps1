param([switch]$Stop,[switch]$Status)
$ErrorActionPreference='Stop'
$taskProjectRoot=Split-Path -Parent $PSScriptRoot
$taskQueue=Join-Path $taskProjectRoot 'variants\upgrade'
if ($Stop) {
    Set-Content -LiteralPath (Join-Path $taskQueue 'stop_queue.request') -Value 'User requested complete checkpoint and GPU release' -Encoding UTF8
    Write-Output 'Requested complete checkpoint and exit; the remaining queue will stop.'
    $taskActive=Get-Content -LiteralPath (Join-Path $taskQueue 'active.json') -Raw | ConvertFrom-Json
    if ($taskActive.step -eq 'train') {
        $taskRequest=Join-Path $taskActive.run 'checkpoints\save_and_stop.request'
        New-Item -ItemType Directory -Path (Split-Path -Parent $taskRequest) -Force | Out-Null
        Set-Content -LiteralPath $taskRequest -Value 'User requested complete checkpoint and exit' -Encoding UTF8
    }
    return
}
& (Join-Path $taskProjectRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'run_upgrade.py') --report-only
Get-Content -LiteralPath (Join-Path $taskQueue 'active.json')
