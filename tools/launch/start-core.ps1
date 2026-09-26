param([string]$ModelRun = '', [switch]$Cpu)
$ErrorActionPreference = 'Stop'
$projectDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (!$ModelRun) {
    $pointer = Join-Path $projectDirectory 'runs/core-active.json'
    if (!(Test-Path -LiteralPath $pointer)) { throw 'Aucun modele actif : utiliser -ModelRun chemin ou terminer une premiere seance.' }
    $ModelRun = (Get-Content -LiteralPath $pointer -Raw | ConvertFrom-Json).run
}
$modelDirectory = [IO.Path]::GetFullPath($ModelRun, $projectDirectory)
if (!(Test-Path -LiteralPath (Join-Path $modelDirectory 'best.json'))) { throw "Checkpoint introuvable : $modelDirectory" }
$sessionId = (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + [guid]::NewGuid().ToString('N').Substring(0,4)
$sessionDirectory = Join-Path $projectDirectory "runs/sessions/core-$sessionId"
New-Item -ItemType Directory -Path $sessionDirectory | Out-Null
$spec = @{project=$projectDirectory; model=$modelDirectory; session=$sessionDirectory; cpu=[bool]$Cpu}
$specPath = Join-Path $sessionDirectory 'session.json'
$spec | ConvertTo-Json | Set-Content -LiteralPath $specPath -Encoding utf8
$scriptPath = Join-Path $PSScriptRoot 'core-session.ps1'
$arguments = @('-NoLogo','-NoProfile','-NoExit','-File',('"'+$scriptPath+'"'),'-SessionFile',('"'+$specPath+'"'))
# L'utilisateur a demande une console visible.
$process = Start-Process -FilePath (Get-Command pwsh.exe).Source -ArgumentList $arguments -WorkingDirectory $projectDirectory -WindowStyle Normal -PassThru
@{pid=$process.Id;session=$sessionDirectory;model=$modelDirectory} | ConvertTo-Json
