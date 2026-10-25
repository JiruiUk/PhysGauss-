param([switch]$Smoke, [double]$Hours=24)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $taskRoot 'reproduction\scripts\enable_cuda.ps1') -ProjectRoot $taskRoot
$taskArgs=@('-u','-m','variants.kinematic_tuning.run','--hours',"$Hours")
if($Smoke){$taskArgs+='--smoke'}
Push-Location $taskRoot
try {
    & (Join-Path $taskRoot 'envs\pidg-cu128\python.exe') @taskArgs
    if($LASTEXITCODE -ne 0){throw "Kinematic coordinator failed: $LASTEXITCODE"}
} finally { Pop-Location }
