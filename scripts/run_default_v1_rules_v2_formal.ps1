[CmdletBinding()]
param(
    [ValidatePattern('^$|^[0-9a-f]{64}$')]
    [string]$ExpectedIdentity = "",
    [switch]$DryRun,
    [switch]$Resume,
    [string]$OutRoot = "",
    [string]$EnvFile = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$sts2CliRoot = (Resolve-Path -LiteralPath (Join-Path $repoRoot "..\sts2-cli") -ErrorAction Stop).Path
$protocolPath = Join-Path $repoRoot "experiments\default_v1_rules_v2_posterior_r10_formal_20260725.json"
$protocol = Get-Content -Raw -Encoding UTF8 -LiteralPath $protocolPath | ConvertFrom-Json

if (-not $OutRoot) {
    $OutRoot = [string]$protocol.output_root
}

$resolvedEnvFile = $null
if (-not $DryRun) {
    if ([string]::IsNullOrWhiteSpace($EnvFile)) {
        throw "-EnvFile is required for a formal run"
    }
    $resolvedEnvFile = (Resolve-Path -LiteralPath $EnvFile -ErrorAction Stop).Path
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONHASHSEED = "0"
$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONSTARTUP -ErrorAction SilentlyContinue
$env:DOTNET_ROLL_FORWARD = "LatestPatch"
$env:STS2_CLI_ROOT = $sts2CliRoot
$env:STS2_LIB = Join-Path $env:STS2_CLI_ROOT "lib"
$env:STS2_GAME_DIR = Join-Path $env:STS2_CLI_ROOT "lib"

Push-Location $repoRoot
try {
    $arguments = @(
        "-B", "-s", "-X", "utf8", "-m", "sts2_bench.default_v1_experiment",
        "--protocol", $protocolPath,
        "--out-root", $OutRoot
    )
    if (-not [string]::IsNullOrWhiteSpace($ExpectedIdentity)) {
        $arguments += @("--expected-identity", $ExpectedIdentity)
    }
    if ($DryRun) {
        $arguments += "--dry-run"
    }
    if ($Resume) {
        $arguments += "--resume"
    }
    if (-not $DryRun) {
        $arguments += @("--env-file", $resolvedEnvFile)
    }
    & python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Default-v1 rules-v2 launcher failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
