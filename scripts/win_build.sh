#!/bin/bash
# This script runs the final PyInstaller build process for Windows.
set -e

# The application version string can be passed as the first argument.
# Defaults to the git tag or a local build identifier.
APP_VERSION=${1:-$(git describe --tags --always || echo "v0.0.0-local")}

# Ensure the MSYS2 environment is configured.
if [ ! -f .msys2_env ]; then
    echo "FATAL: .msys2_env file not found. Please run 'bash scripts/win_setup.sh' first."
    exit 1
fi

# Load the environment variables (MSYS2_PATH, PKG_CONFIG_PATH, etc.)
source .msys2_env

echo "--- Starting Windows Executable Build Process (Version: $APP_VERSION) ---"

# ----------------------------------------------------
# STEP 1: Configure GTK Theme for the bundle
# ----------------------------------------------------
echo "Configuring GTK Theme for bundle..."
mkdir -p etc/gtk-4.0
cat > etc/gtk-4.0/settings.ini <<- EOL
[Settings]
gtk-theme-name=Windows10
gtk-font-name=Segoe UI 9
EOL

# ----------------------------------------------------
# STEP 2: Compile translations
# ----------------------------------------------------
echo "Compiling translations..."
if [ -f "scripts/update_translations.sh" ]; then
    bash scripts/update_translations.sh --compile-only
else
    echo "WARNING: scripts/update_translations.sh not found, skipping translation compile."
fi

# ----------------------------------------------------
# STEP 3: Build with PyInstaller
# ----------------------------------------------------
echo "Building executable with PyInstaller..."

WIN_MSYS2_PATH=$(cygpath -w "$MSYS2_PATH")
echo "Using Windows MSYS2 Path for PyInstaller assets: $WIN_MSYS2_PATH"

# Use Bash parameter expansion `${APP_VERSION#v}` to remove a leading 'v' if it exists.
# This prevents the "vv" issue in the final filename.
CLEAN_VERSION="${APP_VERSION#v}"

pyinstaller --onefile \
  --log-level INFO \
  --name "rayforge-v${CLEAN_VERSION}" \
  --add-data "rayforge/resources;rayforge/resources" \
  --add-data "rayforge/locale;rayforge/locale" \
  --add-data "etc;etc" \
  --add-data "${WIN_MSYS2_PATH}\\mingw64\\share\\glib-2.0\\schemas;glib-2.0\\schemas" \
  --add-data "${WIN_MSYS2_PATH}\\mingw64\\share\\icons;share\\icons" \
  --add-data "${WIN_MSYS2_PATH}\\mingw64\\lib\\girepository-1.0;gi\\repository" \
  --add-binary "${WIN_MSYS2_PATH}\\mingw64\\bin\\libEGL.dll;." \
  --add-binary "${WIN_MSYS2_PATH}\\mingw64\\bin\\libGLESv2.dll;." \
  --add-binary "${WIN_MSYS2_PATH}\\mingw64\\bin\\libvips-42.dll;." \
  --hidden-import "gi._gi_cairo" \
  rayforge/app.py

echo "✅ Executable build complete: dist/rayforge-v${CLEAN_VERSION}.exe"