param(
    [string]$ProjectRoot = 'F:\A_Edinburgh\PhysGauss',
    [string]$BuildPython = ''
)
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $ProjectRoot 'envs\pidg-cu128\python.exe'
if ($BuildPython) { $taskPython = $BuildPython }
$taskPythonRoot = Split-Path -Parent $taskPython
$taskRepo = Join-Path $ProjectRoot 'third_party\PIDG'
$taskVcVars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'PIDG Conda environment is not ready' }
# cmd is used only to obtain the compiler's environment; all file operations use PowerShell/Python.
$taskCompilerEnv = & $env:ComSpec /d /c "`"$taskVcVars`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'MSVC environment activation failed' }
foreach ($taskLine in $taskCompilerEnv) {
    if ($taskLine -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}
$env:CUDA_HOME = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'
$env:TORCH_CUDA_ARCH_LIST = '12.0'
$env:MAX_JOBS = '4'
$env:DISTUTILS_USE_SDK = '1'
$env:MSSdk = '1'
$env:PATH = (Join-Path $taskPythonRoot 'Scripts') + ';' + $env:PATH
$env:TORCH_EXTENSIONS_DIR = Join-Path $ProjectRoot 'reproduction\build-cache'
Set-Location -LiteralPath $taskRepo
$taskWheelDir = Join-Path $ProjectRoot 'reproduction\wheels'
New-Item -ItemType Directory -Path $taskWheelDir -Force | Out-Null
& $taskPython -m pip wheel --no-build-isolation --no-deps -v '.\submodules\simple-knn' -w $taskWheelDir
if ($LASTEXITCODE -ne 0) { throw 'simple-knn build failed' }
& $taskPython -m pip wheel --no-build-isolation --no-deps -v '.\submodules\flow-based-diff-gaussian-rasterization' -w $taskWheelDir
if ($LASTEXITCODE -ne 0) { throw 'Flow rasterizer build failed' }
& $taskPython -c "import hashencoder; print('Official hash encoder and both CUDA wheels built successfully.')"
if ($LASTEXITCODE -ne 0) { throw 'Hash encoder build failed' }
