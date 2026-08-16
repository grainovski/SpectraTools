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
# Written via .NET rather than Set-Content because Windows PowerShell
# 5.1's "-Encoding utf8" means UTF-8 WITH a BOM, and PowerShell 7's
# "utf8NoBOM" does not exist in 5.1 at all. Python's own importer skips a
# leading BOM, so the app never cared -- but the BOM is a real character
# to anything else reading the file as text, and ast.parse() rejects the
# module outright ("invalid non-printable character U+FEFF"), which
# breaks tooling that scans the source tree. The Linux build's heredoc
# has always written this file BOM-less; this makes the two agree.
$buildInfo = @"
VERSION = "$version"
BUILD_DATE = "$buildDate"
"@
[System.IO.File]::WriteAllText(
    "$root/build_info.py",
    $buildInfo + [Environment]::NewLine,
    (New-Object System.Text.UTF8Encoding($false))
)
Write-Host "Stamped build_info.py: VERSION=$version BUILD_DATE=$buildDate"

# --onefile (vs. Linux's --onedir, see build.sh): originally paired with
# Inno Setup convenience and, historically, the now-removed AppImage's
# own AppDir structure on the Linux side. That second reason no longer
# applies (Linux packaging moved to native .deb/.rpm), but onefile is
# kept deliberately for Windows: Inno Setup packages a single exe
# cleanly, and onedir's many loose files would need their own directory
# layout decision in the installer. Tradeoff: onefile's PyInstaller
# bootloader re-extracts the whole bundle to a temp dir on every launch
# (no persistent cache), so Windows cold-starts slower than Linux's
# onedir build does on equivalent hardware, and onefile executables are
# a more common antivirus false-positive target. Revisit if startup
# time or AV false positives become a real user complaint -- see the
# v3.1.0 audit finding that first raised this.
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
