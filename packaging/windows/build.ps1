$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot/../.."
Set-Location $root

if (-not (Test-Path "$root/assets/icon.ico")) {
    & "$root/.venv/Scripts/python.exe" "$root/packaging/make_icon.py"
}

$versionMatch = Select-String -Path "$root/packaging/windows/installer.iss" -Pattern '^AppVersion=(.+)$'
if (-not $versionMatch) {
    throw "Could not find AppVersion in packaging/windows/installer.iss"
}
# Escaped before embedding below: AppVersion is developer-edited (e.g. a
# stray quote from accidentally quoting the value) and gets embedded
# straight into a Python string literal -- an unescaped backslash or
# quote would produce a build_info.py with a Python syntax error.
$version = ($versionMatch.Matches[0].Groups[1].Value -replace '\\', '\\') -replace '"', '\"'
$buildDate = Get-Date -Format "yyyy-MM-dd"
@"
VERSION = "$version"
BUILD_DATE = "$buildDate"
"@ | Set-Content -Path "$root/build_info.py" -Encoding utf8
Write-Host "Stamped build_info.py: VERSION=$version BUILD_DATE=$buildDate"

& "$root/.venv/Scripts/python.exe" -m PyInstaller --noconfirm --onefile --windowed --name SpectraTools --icon "$root/assets/icon.ico" main.py
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$isccCandidates = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$iscc = $isccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    throw "ISCC.exe not found. Install Inno Setup (e.g. 'winget install --id JRSoftware.InnoSetup -e') and retry."
}

& $iscc "packaging/windows/installer.iss"
if ($LASTEXITCODE -ne 0) {
    throw "ISCC.exe failed with exit code $LASTEXITCODE"
}

Write-Host "Installer built in packaging/windows/output/"
