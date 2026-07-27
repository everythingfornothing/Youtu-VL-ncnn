[CmdletBinding()]
param(
    [switch]$FullRegression,
    [ValidateRange(1, 256)]
    [int]$Threads = 8
)

$ErrorActionPreference = "Stop"

$testRoot = $PSScriptRoot
$packageRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $testRoot "..\..")
)
$ucrtBin = Join-Path $testRoot "toolchain\msys64\ucrt64\bin"
$ctest = Join-Path $ucrtBin "ctest.exe"
$runner = Join-Path $testRoot "build\youtu_vl_ncnn.exe"

foreach ($required in @($ctest, $runner)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing Windows test executable: $required"
    }
}

$processPath = [Environment]::GetEnvironmentVariable("Path", "Process")
[Environment]::SetEnvironmentVariable(
    "Path",
    "$ucrtBin;$processPath",
    "Process"
)

& $ctest --test-dir (Join-Path $testRoot "build") --output-on-failure
if ($LASTEXITCODE -ne 0) {
    throw "CTest failed with exit code $LASTEXITCODE"
}

if (-not $FullRegression) {
    return
}

$modelRoot = Join-Path $packageRoot "model"
$testDataRoot = Join-Path $packageRoot "test_data"
$prompt = [string]::Concat(
    [char[]]@(
        0x8FD9,
        0x662F,
        0x4EC0,
        0x4E48,
        0x7C7B,
        0x578B,
        0x7684,
        0x5783,
        0x573E
    )
)
foreach ($tokens in @(32, 64, 128)) {
    $output = Join-Path $testRoot "results\inference-$tokens"
    & $runner `
        --model-root $modelRoot `
        --image (Join-Path $testDataRoot "input_image.jpg") `
        --prompt $prompt `
        --output-dir $output `
        --max-new-tokens $tokens `
        --weight-mode persistent `
        --threads $Threads
    if ($LASTEXITCODE -ne 0) {
        throw "$tokens-token inference failed with exit code $LASTEXITCODE"
    }

    $actual = Join-Path $output "generated_token_ids.npy"
    $reference = Join-Path (
        $testDataRoot
    ) "reference_$tokens\generated_token_ids.npy"
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $actual).Hash
    $referenceHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $reference
    ).Hash
    if ($actualHash -ne $referenceHash) {
        throw "$tokens-token generated IDs differ from the frozen reference"
    }
    Write-Host "$tokens/$tokens generated token IDs: PASS"
}
