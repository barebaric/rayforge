# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['rayforge/app.py'],
    pathex=[],
    binaries=[('C:\\msys64\\mingw64\\bin\\libEGL.dll', '.'), ('C:\\msys64\\mingw64\\bin\\libGLESv2.dll', '.'), ('C:\\msys64\\mingw64\\bin\\libvips-42.dll', '.')],
    datas=[('rayforge/version.txt', 'rayforge'), ('rayforge/resources', 'rayforge/resources'), ('rayforge/locale', 'rayforge/locale'), ('etc', 'etc'), ('C:\\msys64\\mingw64\\share\\glib-2.0\\schemas', 'glib-2.0\\schemas'), ('C:\\msys64\\mingw64\\share\\icons', 'share\\icons'), ('C:\\msys64\\mingw64\\lib\\girepository-1.0', 'gi\\repository')],
    hiddenimports=['gi._gi_cairo'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='rayforge-v4900bbd9',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['rayforge.ico'],
    hide_console='hide-early',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='rayforge-v4900bbd9',
)
