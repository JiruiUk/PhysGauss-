param([double]$Hours=8,[switch]$RenewBudget)
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $taskRoot 'reproduction/scripts/enable_cuda.ps1') -ProjectRoot $taskRoot
$taskArgs=@('-u','-m','variants.integrity_continuation.run','--hours',"$Hours")
if($RenewBudget){$taskArgs+='--renew-budget'}
Push-Location $taskRoot
try { & (Join-Path $taskRoot 'envs/pidg-cu128/python.exe') @taskArgs
    if($LASTEXITCODE -ne 0){throw "Continuation failed: $LASTEXITCODE"}
} finally { Pop-Location }
