#!/bin/bash
# Build Windows executable using the pixi environment.
# Unlike win_build.sh which targets MSYS2, this script uses CONDA_PREFIX
# to locate dependencies installed by pixi.
set -e

APP_VERSION=${1:-$(git describe --tags --always || echo "v0.0.0-local")}
CLEAN_VERSION="${APP_VERSION#v}"
BUNDLE_NAME="rayforge-v${CLEAN_VERSION}"
INSTALLER_EXE_NAME="${BUNDLE_NAME}-installer.exe"

echo "--- Starting Windows Build (pixi) - Version: $APP_VERSION ---"
echo "Embedding version ${APP_VERSION} into rayforge/version.txt"
echo "${APP_VERSION}" > rayforge/version.txt

echo "Creating application icon..."
python scripts/win/win_create_icon.py

echo "Configuring GTK Theme for bundle..."
mkdir -p etc/gtk-4.0
cat > etc/gtk-4.0/settings.ini <<- EOL
[Settings]
gtk-theme-name=Windows10
gtk-font-name=Segoe UI 9
EOL

PREFIX="$CONDA_PREFIX"

SCHEMAS_DIR=$(pkgconf --variable=schemasdir glib-2.0 || echo "${PREFIX}/share/glib-2.0/schemas")
TYPELIB_DIR=$(pkgconf --variable=typelibdir gobject-introspection-1.0 || echo "${PREFIX}/lib/girepository-1.0")
DATADIR=$(pkgconf --variable=datadir gtk4 || echo "${PREFIX}/share")
ICONS_DIR="${DATADIR}/icons"
BIN_DIR="${PREFIX}/bin"

to_win_path() {
    if command -v cygpath &>/dev/null; then
        cygpath -w "$1"
    else
        echo "$1"
    fi
}

WIN_SCHEMAS_DIR=$(to_win_path "$SCHEMAS_DIR")
WIN_ICONS_DIR=$(to_win_path "$ICONS_DIR")
WIN_TYPELIB_DIR=$(to_win_path "$TYPELIB_DIR")
WIN_BIN_DIR=$(to_win_path "$BIN_DIR")

echo "Building with PyInstaller..."
pyinstaller --onedir --hide-console hide-early \
  --log-level INFO \
  --name "${BUNDLE_NAME}" \
  --icon="rayforge.ico" \
  --add-data "rayforge/version.txt;rayforge" \
  --add-data "rayforge/resources;rayforge/resources" \
  --add-data "rayforge/locale;rayforge/locale" \
  --add-data "rayforge/builtin_addons;rayforge/builtin_addons" \
  --add-data "etc;etc" \
  --add-data "${WIN_SCHEMAS_DIR};glib-2.0\\schemas" \
  --add-data "${WIN_ICONS_DIR};share\\icons" \
  --add-data "${WIN_TYPELIB_DIR};gi\\repository" \
  --add-binary "${WIN_BIN_DIR}\\libEGL.dll;." \
  --add-binary "${WIN_BIN_DIR}\\libGLESv2.dll;." \
  --add-binary "${WIN_BIN_DIR}\\libvips-42.dll;." \
  --hidden-import "gi._gi_cairo" \
  --hidden-import "rayforge.core.expression" \
  --hidden-import "rayforge.core.expression.evaluator" \
  --hidden-import "rayforge.core.expression.context" \
  --hidden-import "rayforge.core.expression.errors" \
  --hidden-import "rayforge.core.expression.parser" \
  --hidden-import "rayforge.core.expression.tokenizer" \
  --hidden-import "rayforge.core.expression.validator" \
  --additional-hooks-dir "hooks" \
  rayforge/app.py

echo "PyInstaller build complete: dist/${BUNDLE_NAME}/"

echo "Building installer with NSIS..."
makensis -V2 \
  -DAPP_VERSION="${CLEAN_VERSION}" \
  -DAPP_DIR_NAME="${BUNDLE_NAME}" \
  -DEXECUTABLE_NAME="${BUNDLE_NAME}.exe" \
  -DICON_FILE="rayforge.ico" \
  scripts/win/win_installer.nsi

echo "Build complete: dist/${INSTALLER_EXE_NAME}"
