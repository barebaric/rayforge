#!/usr/bin/env bash
set -euo pipefail

BUNDLE=0
VERSION_OVERRIDE=""
while (($#)); do
    case "$1" in
        --bundle)
            BUNDLE=1
            ;;
        --version)
            VERSION_OVERRIDE="$2"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ ! -f .mac_env ]; then
    echo ".mac_env not found. Run scripts/mac_setup.sh first." >&2
    exit 1
fi

source .mac_env

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to build Rayforge on macOS." >&2
    exit 1
fi

VENV_PATH=${VENV_PATH:-.venv-mac}
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip
python -m pip install --upgrade build pyinstaller
python -m pip install -r requirements.txt

bash scripts/update_translations.sh --compile-only

VERSION=${VERSION_OVERRIDE:-$(git describe --tags --always 2>/dev/null || \
    echo "v0.0.0-local")}
echo "$VERSION" > rayforge/version.txt

python -m build

if (( BUNDLE == 1 )); then
    # Generate macOS icon if it doesn't exist or if SVG is newer
    if [ ! -f "rayforge/resources/icons/icon.icns" ] || \
       [ "website/content/assets/icon.svg" -nt "rayforge/resources/icons/icon.icns" ]; then
        echo "Generating macOS icon..."
        bash scripts/macos_create_icon.sh
    else
        echo "Icon is up to date, skipping generation."
    fi

    pyinstaller --clean --noconfirm Rayforge.spec

    APP_ROOT="dist/Rayforge.app/Contents"
    FW_DIR="$APP_ROOT/Frameworks"
    BIN_DIR="$APP_ROOT/MacOS"

    chmod -R u+w "dist/Rayforge.app" || true

    # Remove conflicting libiconv bundled by cv2.
    rm -f "$FW_DIR/libiconv.2.dylib"

    # Ship critical libs from Homebrew and fix their IDs.
    for lib in \
        libpng16.16.dylib \
        libfontconfig.1.dylib \
        libfreetype.6.dylib \
        libintl.8.dylib \
        libvips.42.dylib \
        libvips-cpp.42.dylib
    do
        if [ -f "/usr/local/lib/$lib" ]; then
            rm -f "$FW_DIR/$lib"
            cp "/usr/local/lib/$lib" "$FW_DIR/"
            install_name_tool -id "@rpath/$lib" "$FW_DIR/$lib"
        fi
    done

    # Fix all library references to use @rpath instead of absolute paths
    echo "Fixing library references..."
    for dylib in "$FW_DIR"/*.dylib; do
        [ -f "$dylib" ] || continue
        # Get all dependencies
        otool -L "$dylib" | grep '/usr/local/' | awk '{print $1}' | while read dep; do
            libname=$(basename "$dep")
            # Only rewrite if we've bundled this library
            if [ -f "$FW_DIR/$libname" ]; then
                install_name_tool -change "$dep" "@rpath/$libname" "$dylib" 2>/dev/null || true
            fi
        done
    done

    # Refresh cv2 dylib symlinks to the parent copies.
    if [ -d "$FW_DIR/cv2/__dot__dylibs" ]; then
        pushd "$FW_DIR/cv2/__dot__dylibs" >/dev/null
        for lib in libpng16.16.dylib libfontconfig.1.dylib \
            libfreetype.6.dylib libintl.8.dylib
        do
            ln -sf ../"$lib" "$lib"
        done
        popd >/dev/null
    fi

    # Note: GTK4 typelibs are automatically bundled by PyInstaller to Resources/gi_typelibs

    # TODO: Bundle vips modules and gdk-pixbuf loaders when vips is installed with SVG support
    # if [ -d "/usr/local/lib/vips-modules-8.17" ]; then
    #     cp -r "/usr/local/lib/vips-modules-8.17" "$FW_DIR/" || true
    # fi
    # if [ -d "/usr/local/lib/gdk-pixbuf-2.0" ]; then
    #     cp -r "/usr/local/lib/gdk-pixbuf-2.0" "$FW_DIR/" || true
    # fi

    # Replace the launcher with a wrapper that sets env vars,
    # keeping the Mach-O as Rayforge.bin.
    if [ -f "$BIN_DIR/Rayforge" ]; then
        mv "$BIN_DIR/Rayforge" "$BIN_DIR/Rayforge.bin"
    fi
    cat > "$BIN_DIR/Rayforge" <<'SH'
#!/bin/bash
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export DYLD_LIBRARY_PATH="$APP_DIR/Frameworks"
export DYLD_FALLBACK_LIBRARY_PATH="$APP_DIR/Frameworks"
export GI_TYPELIB_PATH="$APP_DIR/Resources/gi_typelibs"
export GIO_EXTRA_MODULES="$APP_DIR/Frameworks/gio_modules"
exec "$APP_DIR/MacOS/Rayforge.bin" "$@"
SH
    chmod +x "$BIN_DIR/Rayforge"

    # Ensure rpath points to the bundled Frameworks.
    install_name_tool -add_rpath @executable_path/../Frameworks \
        "$BIN_DIR/Rayforge.bin" 2>/dev/null || true

    # Make sure the plist still points to the wrapper.
    /usr/libexec/PlistBuddy -c "Set :CFBundleExecutable Rayforge" \
        "$APP_ROOT/Info.plist" 2>/dev/null || true
fi

echo "Build artifacts created in dist/ and dist/*.whl"

deactivate
