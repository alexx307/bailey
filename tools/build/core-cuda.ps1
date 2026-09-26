param([string[]]$CargoArguments = @('build', '-p', 'bailey-core', '--features', 'cuda'))
$ErrorActionPreference = 'Stop'
$projectDirectory = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$vcvars = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$cudaDirectory = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
if (!(Test-Path -LiteralPath $vcvars)) { throw "MSVC introuvable : $vcvars" }
if (!(Test-Path -LiteralPath (Join-Path $cudaDirectory 'bin/nvcc.exe'))) { throw 'CUDA 13.0 introuvable.' }
# Importer uniquement l'environnement de compilation dans ce processus.
$compilerEnvironment = & cmd.exe /d /s /c ('"' + $vcvars + '" >nul && set')
if ($LASTEXITCODE -ne 0) { throw "Initialisation de MSVC impossible." }
$names = @('PATH','INCLUDE','LIB','LIBPATH','VCINSTALLDIR','VCToolsInstallDir','WindowsSdkDir','WindowsSDKVersion','VSINSTALLDIR')
foreach ($line in $compilerEnvironment) {
    $parts = $line -split '=', 2
    if ($parts.Count -eq 2 -and $parts[0] -in $names) {
        [Environment]::SetEnvironmentVariable($parts[0], $parts[1], 'Process')
    }
}
$env:CUDA_PATH = $cudaDirectory
$env:PATH = (Join-Path $cudaDirectory 'bin') + ';' + $env:PATH
Push-Location -LiteralPath $projectDirectory
try {
    & cargo @CargoArguments
    $buildExitCode = $LASTEXITCODE
} finally { Pop-Location }
if ($buildExitCode -ne 0) { throw "Cargo a echoue (code $buildExitCode)." }
