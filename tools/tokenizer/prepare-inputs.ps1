param(
    [string]$Out = 'data/tokenizer-inputs-generalist-v1',
    [string]$French = 'data/wikipedia-fr-curated-v1',
    [string]$English = 'data/wikipedia-en-tokenizer-curated-v1',
    [string]$Original = 'assets/tokenizer/corpus-v1',
    [long]$MaxTextBytes = 8388608
)
$ErrorActionPreference = 'Stop'
$projectDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$destination = [IO.Path]::GetFullPath((Join-Path $projectDirectory $Out))
if (Test-Path -LiteralPath $destination) { throw 'Utiliser un nouveau dossier de sources.' }
$frenchDirectory = [IO.Path]::GetFullPath((Join-Path $projectDirectory $French))
$englishDirectory = [IO.Path]::GetFullPath((Join-Path $projectDirectory $English))
$originalDirectory = [IO.Path]::GetFullPath((Join-Path $projectDirectory $Original))
$originalManifest = Join-Path $originalDirectory 'manifest.json'
$originalSpec = Get-Content -LiteralPath $originalManifest -Raw | ConvertFrom-Json
foreach ($directory in @($frenchDirectory, $englishDirectory)) {
    foreach ($name in @('train.txt', 'manifest.json')) {
        if (!(Test-Path -LiteralPath (Join-Path $directory $name) -PathType Leaf)) {
            throw "Source absente : $directory/$name"
        }
    }
}
$codeFiles = foreach ($subject in @('model', 'training', 'inference', 'web', 'knowledge', 'research', 'app', 'forge')) {
    Get-ChildItem -LiteralPath (Join-Path $projectDirectory "crates/bailey-core/src/$subject") -Filter '*.rs' -File -Recurse |
        Where-Object {
            $_.Name -notlike '*test*' -and
            (Get-Content -LiteralPath $_.FullName -Raw) -notmatch '#\[cfg\(test\)\]'
        }
}
New-Item -ItemType Directory -Path $destination | Out-Null
$sources = [Collections.Generic.List[object]]::new()
function ConvertTo-RelativePath([string]$Base, [string]$Target) {
    $baseUri = New-Object System.Uri(($Base.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar))
    $targetUri = New-Object System.Uri($Target)
    [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString())
}
function Write-Utf8NoBom([string]$Path, [string]$Content) {
    [IO.File]::WriteAllText($Path, $Content, (New-Object System.Text.UTF8Encoding($false)))
}
function Add-MixSource([string]$Id, [string]$Domain, [string]$File, [string]$Provenance) {
    $sources.Add([ordered]@{
        id = $Id; domain = $Domain; partition = 'train'
        file = (ConvertTo-RelativePath $destination $File).Replace('\','/')
        sha256 = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant()
        provenance = (ConvertTo-RelativePath $destination $Provenance).Replace('\','/')
    })
}
Add-MixSource 'wikipedia-fr-train' 'french' (Join-Path $frenchDirectory 'train.txt') (Join-Path $frenchDirectory 'manifest.json')
Add-MixSource 'wikipedia-en-train' 'english' (Join-Path $englishDirectory 'train.txt') (Join-Path $englishDirectory 'manifest.json')
foreach ($item in $originalSpec.files) {
    if ($item.category -notin @('english', 'code')) { throw 'Categorie originale inconnue.' }
    $file = [IO.Path]::GetFullPath((Join-Path $originalDirectory $item.path))
    if (!$file.StartsWith($originalDirectory + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Chemin de source originale hors corpus.'
    }
    Add-MixSource ('original-' + $item.path) $item.category $file $originalManifest
}
$codeManifest = Join-Path $destination 'code-source-manifest.json'
$codeRecords = foreach ($file in ($codeFiles | Sort-Object FullName)) {
    $relative = ConvertTo-RelativePath $projectDirectory $file.FullName
    $target = Join-Path $destination (Join-Path 'code' $relative)
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $target
    Add-MixSource ('bailey-' + $relative.Replace('\','/')) 'code' $target $codeManifest
    [ordered]@{source=$relative;sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$revision = & git -C $projectDirectory rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Revision Git indisponible.' }
$codeManifestJson = [ordered]@{
    origin = 'Local Bailey working-copy source snapshot, never uploaded'
    revision_context = $revision
    license = 'Local project sources; no public redistribution license asserted'
    selection = 'Named implementation folders only; exclude filenames containing test and files with cfg(test); exclude tokenization, probes and curriculum'
    files = @($codeRecords)
} | ConvertTo-Json -Depth 8
Write-Utf8NoBom $codeManifest $codeManifestJson
$mixJson = [ordered]@{
    seed = 42; max_text_bytes = $MaxTextBytes; percentages = @(70,15,15); sources = @($sources)
} | ConvertTo-Json -Depth 8
Write-Utf8NoBom (Join-Path $destination 'mix.json') $mixJson
Write-Output "Configuration : $destination/mix.json ; $($sources.Count) sources explicites."
