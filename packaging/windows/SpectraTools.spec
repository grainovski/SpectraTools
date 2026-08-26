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

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SpectraTools',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,
    windowed=True,
    icon=_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SpectraTools',
)
