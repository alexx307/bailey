$ErrorActionPreference = 'Stop'
$projectDirectory = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
Push-Location -LiteralPath $projectDirectory
try {
    $oversized = Get-ChildItem -LiteralPath crates -Filter '*.rs' -Recurse -File |
        Where-Object { (Get-Content -LiteralPath $_.FullName).Count -gt 300 }
    if ($oversized) {
        throw "Fichiers Rust de plus de 300 lignes : $($oversized.FullName -join ', ')"
    }
    cargo fmt --all --check
    if ($LASTEXITCODE -ne 0) { throw 'Le formatage doit etre corrige.' }
    cargo clippy --workspace --all-targets -- -D warnings
    if ($LASTEXITCODE -ne 0) { throw "L'analyse statique a echoue." }
    cargo test --workspace
    if ($LASTEXITCODE -ne 0) { throw 'Les tests ont echoue.' }
} finally {
    Pop-Location
}
