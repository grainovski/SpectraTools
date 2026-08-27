# SpectraTools.spec — PyInstaller onedir build for Windows
# Run from project root: python -m PyInstaller --noconfirm packaging/windows/SpectraTools.spec
#
# --collect-all awkward_cpp is REQUIRED: uproot depends on awkward, whose
# awkward_cpp component loads awkward-cpu-kernels.dll via ctypes.  PyInstaller's
# static analysis never sees a ctypes load, so without this the build succeeds
# but the app crashes at import with "Failed to load dynlib/dll".

import os
from PyInstaller.utils.hooks import collect_all

_root = os.path.abspath(os.path.join(SPECPATH, '..', '..'))
_icon = os.path.join(_root, 'assets', 'icon.ico')

datas_awk, binaries_awk, hiddenimports_awk = collect_all('awkward_cpp')

a = Analysis(
    [os.path.join(_root, 'main.py')],
    pathex=[_root],
    binaries=binaries_awk,
    datas=datas_awk,
    hiddenimports=hiddenimports_awk,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'wx', 'gtk', 'gi',
        'IPython', 'notebook',
        'sklearn', 'cv2',
        'docutils', 'sphinx',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

# UPX packing is switched OFF here, deliberately. UPX compresses each
# executable and DLL, and a packed binary is one of the oldest antivirus
# heuristics there is -- self-extracting code is what packers and droppers
# both look like. That matters more than usual here: SpectraTools is
# unsigned, so scanners have nothing but shape to judge it on, and a user
# did hit Windows refusing the unsigned installer (INSTALL.md exists
# because of it). Moving from onefile to onedir removed exactly that
# self-extracting signature; packing the pieces would hand it straight
# back, for a few MB, against an installer that already compresses with
# lzma2/solid. It was inert when written -- UPX is not installed on the
# build host, so PyInstaller skipped it silently and the shipped v4.2.1
# binaries were never packed -- but 'inert until someone installs a tool'
# is a trap, not a setting.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SpectraTools',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    console=False,
    windowed=True,
    icon=_icon,
)

# UPX packing is switched OFF here, deliberately. UPX compresses each
# executable and DLL, and a packed binary is one of the oldest antivirus
# heuristics there is -- self-extracting code is what packers and droppers
# both look like. That matters more than usual here: SpectraTools is
# unsigned, so scanners have nothing but shape to judge it on, and a user
# did hit Windows refusing the unsigned installer (INSTALL.md exists
# because of it). Moving from onefile to onedir removed exactly that
# self-extracting signature; packing the pieces would hand it straight
# back, for a few MB, against an installer that already compresses with
# lzma2/solid. It was inert when written -- UPX is not installed on the
# build host, so PyInstaller skipped it silently and the shipped v4.2.1
# binaries were never packed -- but 'inert until someone installs a tool'
# is a trap, not a setting.
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SpectraTools',
)
