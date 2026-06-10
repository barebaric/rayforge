# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['gi._gi_cairo', 'cairosvg']
hiddenimports += collect_submodules('rayforge.ui_gtk.canvas2d')
hiddenimports += collect_submodules('rayforge.ui_gtk.canvas2d.elements')
hiddenimports += collect_submodules('rayforge.ui_gtk.shared')
hiddenimports += collect_submodules('rayforge.image')
hiddenimports += collect_submodules('rayforge.core')
hiddenimports.append('rayforge.ui_gtk.canvas2d.elements.workpiece')

# Use modern .icon (via Assets.car) when available, fall back to .icns.
_use_car = os.path.exists('Assets.car')
_icon = None if _use_car else 'rayforge.icns'

_datas = [
    ('rayforge/version.txt', 'rayforge'),
    ('rayforge/resources', 'rayforge/resources'),
    ('rayforge/locale', 'rayforge/locale'),
    ('rayforge/builtin_addons', 'rayforge/builtin_addons'),
]
if _use_car:
    _datas.append(('Assets.car', '.'))

a = Analysis(
    ['rayforge/app.py'],
    pathex=['.'],
    binaries=[],
    datas=_datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
    hooksconfig={
        'gi': {
            'module-versions': {
                'Gtk': '4.0',
                'Adw': '1',
            },
        },
    },
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
    icon=[_icon] if _icon else [],
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
    icon=_icon,
    bundle_identifier='org.rayforge.rayforge',
    info_plist={'CFBundleIconName': 'rayforge'} if _use_car else {},
)
