param([string]$ProjectRoot = 'F:\A_Edinburgh\PhysGauss')
$ErrorActionPreference = 'Stop'
$taskVcVars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$env:VSLANG = '1033'
$taskCompilerEnv = & $env:ComSpec /d /c "`"$taskVcVars`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'MSVC environment activation failed' }
foreach ($taskLine in $taskCompilerEnv) {
    if ($taskLine -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
$taskEnvironment = Join-Path $ProjectRoot 'envs\pidg-cu128'
$env:PATH = "$taskEnvironment;$taskEnvironment\Scripts;$taskEnvironment\Library\bin;" + $env:PATH
$env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:TORCH_CUDA_ARCH_LIST = '12.0'
$env:MAX_JOBS = '4'
$env:DISTUTILS_USE_SDK = '1'
$env:MSSdk = '1'
$env:PYTHONUNBUFFERED = '1'
$env:TORCH_HOME = Join-Path $ProjectRoot 'reproduction\torch-cache'
$env:TORCH_EXTENSIONS_DIR = Join-Path $ProjectRoot 'reproduction\build-cache'
