param([switch]$Stop,[switch]$Status)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskRuns=Join-Path $taskRoot 'runs\pidg_constitutive'
if ($Stop) {
    Set-Content -LiteralPath (Join-Path $taskRuns 'stop_queue.request') -Value 'User requested full checkpoint and exit' -Encoding UTF8
    Write-Output 'Save-and-exit requested. The live controller forwards it once; verify process exit and checkpoint before claiming GPU release.'
} else {
    & (Join-Path $taskRoot 'envs\pidg-cu128\python.exe') (Join-Path $PSScriptRoot 'run_constitutive_matrix.py') --report-only
    Get-Content -LiteralPath (Join-Path $taskRuns 'controller.json')
    Get-Content -LiteralPath (Join-Path $taskRuns 'active.json')
}
