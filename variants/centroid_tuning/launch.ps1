param([switch]$Smoke,[double]$Hours=24,[ValidateSet('original','v3')][string]$Base='original')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $taskRoot 'reproduction/scripts/enable_cuda.ps1') -ProjectRoot $taskRoot
$taskArgs=@('-u','-m','variants.centroid_tuning.run','--hours',"$Hours",'--base',$Base)
if($Smoke){$taskArgs+='--smoke'}
Push-Location $taskRoot
try {
    & (Join-Path $taskRoot 'envs/pidg-cu128/python.exe') @taskArgs
    if($LASTEXITCODE -ne 0){throw "Centroid experiment failed: $LASTEXITCODE"}
} finally { Pop-Location }
