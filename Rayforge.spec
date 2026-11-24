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

# File types Rayforge can open. The first entry owns the project
# extensions, so launching a project never asks for an importer.
_document_types = [
    {
        'CFBundleTypeName': 'Rayforge Project',
        'CFBundleTypeRole': 'Editor',
        'LSHandlerRank': 'Owner',
        'CFBundleTypeExtensions': ['ryp', 'rfs'],
    },
    {
        'CFBundleTypeName': 'Ruida Laser Cutter File',
        'CFBundleTypeRole': 'Viewer',
        'LSHandlerRank': 'Owner',
        'CFBundleTypeExtensions': ['rd'],
    },
    {
        'CFBundleTypeName': 'G-code File',
        'CFBundleTypeRole': 'Viewer',
        'LSHandlerRank': 'Alternate',
        'CFBundleTypeExtensions': [
            'gcode', 'ngc', 'nc', 'tap', 'gc',
        ],
        'CFBundleTypeMIMETypes': ['text/x.gcode'],
    },
    {
        'CFBundleTypeName': 'LightBurn Project File',
        'CFBundleTypeRole': 'Viewer',
        'LSHandlerRank': 'Alternate',
        'CFBundleTypeExtensions': ['lbrn', 'lbrn2'],
        'CFBundleTypeMIMETypes': ['application/x-lightburn'],
    },
    {
        'CFBundleTypeName': 'HPGL Plot File',
        'CFBundleTypeRole': 'Viewer',
        'LSHandlerRank': 'Alternate',
        'CFBundleTypeExtensions': ['plt', 'hpgl', 'hpg', 'hgl'],
        'CFBundleTypeMIMETypes': ['application/vnd.hp-hpgl'],
    },
    {
        'CFBundleTypeName': 'Vector Drawing',
        'CFBundleTypeRole': 'Viewer',
        'LSHandlerRank': 'Alternate',
        'CFBundleTypeExtensions': [
            'dxf', 'cdr', 'svg', 'svgz', 'pdf', 'ai', 'eps', 'epsf', 'ps',
        ],
        'CFBundleTypeMIMETypes': [
            'image/vnd.dxf', 'application/vnd.corel-draw',
            'image/svg+xml', 'application/pdf',
        ],
    },
    {
        'CFBundleTypeName': 'Raster Image',
        'CFBundleTypeRole': 'Viewer',
        'LSHandlerRank': 'Alternate',
        'CFBundleTypeOSTypes': ['PNG ', 'JPEG', 'GIFf', 'TIFF', 'BMP '],
        'CFBundleTypeExtensions': [
            'png', 'jpg', 'jpeg', 'gif', 'tif', 'tiff', 'bmp', 'webp',
        ],
        'CFBundleTypeMIMETypes': [
            'image/png', 'image/jpeg', 'image/gif', 'image/tiff',
            'image/bmp', 'image/webp',
        ],
    },
]

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
    # Pillow is not used by the application; it only appears in the
    # module graph through optional imports of dependencies. Bundling
    # it drags in PIL/.dylibs, whose harfbuzz shadows the GTK4 one and
    # crashes the app at startup.
    excludes=['PIL'],
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
    info_plist={
        **({'CFBundleIconName': 'rayforge'} if _use_car else {}),
        'CFBundleDocumentTypes': _document_types,
        'LSMinimumSystemVersion': '12.0',
        # Without this key, macOS TCC kills the process the moment any
        # camera is opened (even just enumerating devices), so it must
        # stay in sync with the camera sources in rayforge/camera.
        'NSCameraUsageDescription': (
            'Rayforge uses the camera to capture your work surface, '
            'so it can align your design with the material.'
        ),
    },
)
