param(
    [string]$ModelRun = '',
    [ValidateRange(1,100000)][int]$Steps = 800,
    [string]$Out = '',
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$projectDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
Push-Location -LiteralPath $projectDirectory
try {
    if (!$ModelRun) { $ModelRun = (Get-Content -LiteralPath 'runs/core-active.json' -Raw | ConvertFrom-Json).run }
    $modelDirectory = [IO.Path]::GetFullPath($ModelRun, $projectDirectory)
    if (!(Test-Path -LiteralPath (Join-Path $modelDirectory 'best.json'))) { throw 'Checkpoint initial introuvable.' }
    if (!$Out) { $Out = 'runs/dialogue-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') }
    $sessionDirectory = [IO.Path]::GetFullPath($Out, $projectDirectory)
    if (Test-Path -LiteralPath $sessionDirectory) { throw 'Utiliser un nouveau dossier de seance.' }
    if (!$SkipBuild) { & ./tools/build/core-cuda.ps1 }
    $env:PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;' + $env:PATH
    New-Item -ItemType Directory -Path $sessionDirectory | Out-Null
    $executable = Join-Path $sessionDirectory 'bailey-core.exe'
    Copy-Item -LiteralPath 'target/debug/bailey-core.exe' -Destination $executable
    $datasetDirectory = Join-Path $sessionDirectory 'data'
    & $executable prepare-dialogue --out $datasetDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Preparation du cours impossible.' }
    $runDirectory = Join-Path $sessionDirectory 'model'
    $warmup = [Math]::Min(40, [Math]::Floor($Steps / 10))
    $evaluation = [Math]::Min(100, $Steps)
    Write-Host "Entrainement CUDA : $runDirectory" -ForegroundColor Cyan
    Write-Host 'STOP dans ce dossier arrete la seance entre deux etapes. Aucun modele actif ne sera remplace automatiquement.'
    & $executable --device cuda train --init-from $modelDirectory --tokenizer (Join-Path $modelDirectory 'tokenizer.json') `
        --data $datasetDirectory --out $runDirectory --steps $Steps --sequence 128 --batch-size 8 --objective dialogue `
        --learning-rate 0.0001 --warmup-steps $warmup --min-lr-ratio 0.1 --max-grad-norm 1 `
        --evaluation-windows 28 --eval-every $evaluation
    if ($LASTEXITCODE -ne 0) { throw 'Entrainement interrompu par une erreur.' }
    if (Test-Path -LiteralPath (Join-Path $runDirectory 'interrupted.json')) { Write-Host 'Seance arretee, poids precedents conserves.'; return }
    & $executable --device cuda dialogue-report --run $runDirectory `
        --prompts assets/curricula/french-dialogue-v1/validation.json --out (Join-Path $sessionDirectory 'dialogue-validation.json')
    if ($LASTEXITCODE -ne 0) { throw 'Generation du rapport impossible.' }
    Write-Host "Rapport disponible dans $sessionDirectory. Ce petit cours ne constitue pas un preentrainement generaliste."
} finally { Pop-Location }
