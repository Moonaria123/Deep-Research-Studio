[CmdletBinding()]
param(
    [string]$Python = "py",
    [string]$Wheelhouse = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$environmentRoot = Join-Path $repoRoot ".tmp\desktop-sidecar-env"
$environmentPython = Join-Path $environmentRoot "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $environmentPython -PathType Leaf)) {
    $launcherName = [System.IO.Path]::GetFileNameWithoutExtension($Python)
    if ($launcherName -eq "py") {
        & $Python -3.12 -m venv $environmentRoot
    } else {
        & $Python -m venv $environmentRoot
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the isolated Python 3.12 build environment."
    }
}

$installArguments = @(
    "-m", "pip", "install", "--disable-pip-version-check",
    "-r", (Join-Path $repoRoot "apps\desktop\requirements-sidecar.txt"),
    "-r", (Join-Path $repoRoot "apps\desktop\requirements-build.txt")
)
if ($Wheelhouse) {
    $resolvedWheelhouse = (Resolve-Path -LiteralPath $Wheelhouse).Path
    $installArguments += @("--find-links", $resolvedWheelhouse)
}
& $environmentPython @installArguments
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the pinned sidecar build requirements."
}

& $environmentPython (Join-Path $PSScriptRoot "build_sidecar.py") --python $environmentPython
if ($LASTEXITCODE -ne 0) {
    throw "The sidecar build failed."
}
