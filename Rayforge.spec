# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['rayforge/app.py'],
    pathex=[],
    binaries=[],
    datas=[('rayforge/version.txt', 'rayforge'), ('rayforge/resources', 'rayforge/resources'), ('rayforge/locale', 'rayforge/locale')],
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
    name='Rayforge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Rayforge',
)
app = BUNDLE(
    coll,
    name='Rayforge.app',
    icon=None,
    bundle_identifier='org.rayforge.rayforge',
)
