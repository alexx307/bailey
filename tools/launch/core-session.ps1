param([Parameter(Mandatory)][string]$SessionFile)
$ErrorActionPreference = 'Stop'
$spec = Get-Content -LiteralPath $SessionFile -Raw | ConvertFrom-Json
Set-Location -LiteralPath $spec.project
$settings = Get-Content -LiteralPath (Join-Path $spec.model 'config.json') -Raw | ConvertFrom-Json
$label = if ($settings.objective -eq 'dialogue') { 'cours de francais' } else { 'fondation de langage' }
$Host.UI.RawUI.WindowTitle = "Bailey Core | $label"
function Status([string]$stage) {
    @{stage=$stage;updated=(Get-Date).ToString('o')} | ConvertTo-Json |
        Set-Content -LiteralPath (Join-Path $spec.session 'status.json') -Encoding utf8
}
Start-Transcript -LiteralPath (Join-Path $spec.session 'terminal.log') | Out-Null
try {
    Status 'building'
    if ($spec.cpu) {
        if (!$spec.skipBuild) {
            & cargo build
            if ($LASTEXITCODE -ne 0) { throw 'Compilation CPU impossible.' }
        }
        $device = 'cpu'
    } else {
        if (!$spec.skipBuild) { & .\tools\build\core-cuda.ps1 }
        $env:PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;' + $env:PATH
        $device = 'cuda'
    }
    $exe = Join-Path $spec.session 'bailey-core.exe'
    Copy-Item -LiteralPath (Join-Path $spec.project 'target/debug/bailey-core.exe') -Destination $exe
    Write-Host 'BAILEY CORE - fondation de langage' -ForegroundColor Cyan
    & $exe info
    Write-Host 'Le petit corpus initial verifie le moteur. Il ne constitue pas encore un preentrainement du francais.' -ForegroundColor Yellow
    Status 'console'
    & $exe --device $device console --run $spec.model
    if ($LASTEXITCODE -ne 0) { throw 'Console interrompue par une erreur.' }
    Status 'closed'
} catch { Status 'failed'; Write-Host $_.Exception.Message -ForegroundColor Red }
finally { Stop-Transcript | Out-Null }
