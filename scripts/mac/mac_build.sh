#!/usr/bin/env bash
set -euo pipefail

DO_BUILD=0
DO_BUNDLE=0
DO_DMG=0
DO_RUN_APP=0
VERSION_OVERRIDE=""
GREEN="\033[0;32m"
NC="\033[0m"

print_info() {
    local title=$1
    printf "${GREEN}%s${NC}\n" "$title"
}

while (($#)); do
    case "$1" in
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

echo ""
print_info "======================================"
print_info "    Rayforge macOS Build Script"
print_info "======================================"
echo ""
echo "Select build option:"
echo "    1) Build"
echo "    2) Bundle (.app)"
echo "    3) Run bundled app"
echo "    4) Distribution package (.dmg)"
echo "    5) All of the above"
echo "    6) exit"
echo ""
read -r -p "Choice (1-6): " BUILD_CHOICE
case "$BUILD_CHOICE" in
    1)
        DO_BUILD=1
        ;;
    2)
        DO_BUNDLE=1
        ;;
    3)
        DO_RUN_APP=1
        ;;
    4)
        DO_DMG=1
        ;;
    5)
        DO_BUILD=1
        DO_BUNDLE=1
        DO_DMG=1
        ;;
    6)
        exit 0
        ;;
    *)
        exit 0
        ;;
esac

if (( DO_RUN_APP == 1 )); then
    APP_BIN="./dist/Rayforge.app/Contents/MacOS/Rayforge"
    if [ ! -x "$APP_BIN" ]; then
        echo "$APP_BIN not found or not executable." >&2
        echo "Build the app bundle first (option 2)." >&2
        exit 1
    fi
    "$APP_BIN"
    exit 0
fi

if [ ! -f .mac_env ]; then
    echo ".mac_env not found. Run scripts/mac/mac_setup.sh first." >&2
    exit 1
fi

source .mac_env

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to build Rayforge on macOS." >&2
    exit 1
fi

echo ""
echo ""
print_info "  Environment Setup"
print_info "--------------------------------------"
echo ""

VENV_PATH=${VENV_PATH:-.venv}
PYTHON_BOOTSTRAP=python3
if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BOOTSTRAP=python3.11
fi
if [ ! -d "$VENV_PATH" ]; then
    "$PYTHON_BOOTSTRAP" -m venv "$VENV_PATH"
fi

VENV_PY="$VENV_PATH/bin/python"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install --upgrade build pyinstaller
TMP_REQUIREMENTS=$(mktemp)
grep -Evi '^(PyOpenGL_accelerate|opencv[_-]python)' \
    requirements.txt > "$TMP_REQUIREMENTS"

if [ "$(uname -s)" = "Darwin" ]; then
    awk '
        BEGIN { done_numpy = 0; done_scipy = 0 }
        /^numpy==/ {
            print "numpy==1.26.4"
            done_numpy = 1
            next
        }
        /^scipy==/ {
            print "scipy==1.11.4"
            done_scipy = 1
            next
        }
        { print }
        END {
            if (done_numpy == 0) {
                print "numpy==1.26.4"
            }
            if (done_scipy == 0) {
                print "scipy==1.11.4"
            }
        }
    ' "$TMP_REQUIREMENTS" > "$TMP_REQUIREMENTS.patched"
    mv "$TMP_REQUIREMENTS.patched" "$TMP_REQUIREMENTS"
fi

if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "x86_64" ]; then
    echo "Installing pinned OpenCV wheel for macOS x86_64..."
    "$VENV_PY" -m pip install --only-binary=:all: \
        "opencv-python==4.10.0.84"
fi

"$VENV_PY" -m pip install -r "$TMP_REQUIREMENTS"
if [ "$(uname -s)" = "Darwin" ]; then
    "$VENV_PY" -m pip install --upgrade --force-reinstall \
        "numpy==1.26.4" "scipy==1.11.4"
fi
rm -f "$TMP_REQUIREMENTS"
"$VENV_PY" -m pip install PyOpenGL_accelerate==3.1.10 || \
    echo "PyOpenGL_accelerate install failed; continuing."

bash scripts/update_translations.sh --compile-only

VERSION=${VERSION_OVERRIDE:-$(git describe --tags --always 2>/dev/null || \
    echo "v0.0.0-local")}
echo "$VERSION" > rayforge/version.txt

if (( DO_BUILD == 1 )); then
    echo ""
    echo ""
    print_info "  Build"
    print_info "--------------------------------------"
    echo ""
    "$VENV_PY" -m build
elif [ -d "dist/Rayforge.app" ] && (( DO_BUNDLE == 0 )); then
    echo "Note: dist/Rayforge.app exists but was not rebuilt." >&2
fi

if (( DO_BUNDLE == 1 )); then
    echo ""
    echo ""
    print_info "  .app Bundle"
    print_info "--------------------------------------"
    echo ""
    python - <<'PY'
import os
import shutil
import stat
from pathlib import Path

def _onerror(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        func(path)
    except Exception:
        pass

for target in ("dist/Rayforge", "dist/Rayforge.app"):
    path = Path(target)
    if path.exists():
        shutil.rmtree(path, onerror=_onerror)
PY
    # Generate macOS icon if it doesn't exist or if SVG is newer
    if [ ! -f "rayforge.icns" ] || \
       [ "website/static/assets/icon-app.svg" -nt "rayforge.icns" ]; then
        echo "Generating macOS icon..."
        bash scripts/mac/mac_create_icon.sh
    else
        echo "Icon is up to date, skipping generation."
    fi

    "$VENV_PY" -m PyInstaller --clean --noconfirm Rayforge.spec

    APP_ROOT="dist/Rayforge.app/Contents"
    FW_DIR="$APP_ROOT/Frameworks"
    BIN_DIR="$APP_ROOT/MacOS"

    chmod -R u+w "dist/Rayforge.app" || true

    # Remove conflicting libiconv bundled by cv2.
    rm -f "$FW_DIR/libiconv.2.dylib"

    # Replace the launcher with a wrapper that sets env vars,
    # keeping the Mach-O as Rayforge.bin.
    if [ -f "$BIN_DIR/Rayforge" ] && [ ! -f "$BIN_DIR/Rayforge.bin" ]; then
        if file "$BIN_DIR/Rayforge" | grep -q "Mach-O"; then
            mv "$BIN_DIR/Rayforge" "$BIN_DIR/Rayforge.bin"
        else
            cp "$BIN_DIR/Rayforge" "$BIN_DIR/Rayforge.bin"
        fi
    fi
    if [ -f "$BIN_DIR/Rayforge.bin" ]; then
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
        install_name_tool -add_rpath @executable_path/../Frameworks \
            "$BIN_DIR/Rayforge.bin" 2>/dev/null || true
    fi

    BREW_PREFIX=""
    if command -v brew >/dev/null 2>&1; then
        BREW_PREFIX=$(brew --prefix)
    fi
    if [ -z "$BREW_PREFIX" ]; then
        if [ -d "/opt/homebrew" ]; then
            BREW_PREFIX="/opt/homebrew"
        else
            BREW_PREFIX="/usr/local"
        fi
    fi

    # Ship critical libs from Homebrew and fix their IDs.
    for lib in \
        libpng16.16.dylib \
        libsharpyuv.0.dylib \
        libfontconfig.1.dylib \
        libfreetype.6.dylib \
        libintl.8.dylib \
        libvips.42.dylib \
        libvips-cpp.42.dylib \
        libOpenEXR-3_4.33.dylib \
        libOpenEXRCore-3_4.33.dylib \
        libIex-3_4.33.dylib \
        libIlmThread-3_4.33.dylib \
        libImath-3_2.30.dylib \
        libarchive.13.dylib \
        libcfitsio.10.dylib \
        libexif.12.dylib \
        libfftw3.3.dylib \
        libhwy.1.dylib \
        libopenjp2.7.dylib
    do
        if [ -f "$BREW_PREFIX/lib/$lib" ]; then
            rm -f "$FW_DIR/$lib"
            cp "$BREW_PREFIX/lib/$lib" "$FW_DIR/"
            install_name_tool -id "@rpath/$lib" "$FW_DIR/$lib"
        fi
    done
    copy_keg_lib() {
        local libname=$1
        shift
        if [ -f "$FW_DIR/$libname" ]; then
            return
        fi
        for lib_dir in "$@"; do
            if [ -f "$lib_dir/$libname" ]; then
                rm -f "$FW_DIR/$libname"
                cp "$lib_dir/$libname" "$FW_DIR/"
                install_name_tool -id "@rpath/$libname" "$FW_DIR/$libname"
                break
            fi
        done
    }
    copy_keg_lib libfontconfig.1.dylib \
        "$BREW_PREFIX/opt/fontconfig/lib" \
        "/usr/local/opt/fontconfig/lib" \
        "/opt/homebrew/opt/fontconfig/lib"
    copy_keg_lib libfreetype.6.dylib \
        "$BREW_PREFIX/opt/freetype/lib" \
        "/usr/local/opt/freetype/lib" \
        "/opt/homebrew/opt/freetype/lib"
    copy_keg_lib libsharpyuv.0.dylib \
        "$BREW_PREFIX/opt/webp/lib" \
        "/usr/local/opt/webp/lib" \
        "/opt/homebrew/opt/webp/lib"
    copy_keg_lib libintl.8.dylib \
        "$BREW_PREFIX/opt/gettext/lib" \
        "/usr/local/opt/gettext/lib" \
        "/opt/homebrew/opt/gettext/lib"
    copy_keg_lib libOpenEXR-3_4.33.dylib \
        "$BREW_PREFIX/opt/openexr/lib" \
        "/usr/local/opt/openexr/lib" \
        "/opt/homebrew/opt/openexr/lib"
    copy_keg_lib libOpenEXRCore-3_4.33.dylib \
        "$BREW_PREFIX/opt/openexr/lib" \
        "/usr/local/opt/openexr/lib" \
        "/opt/homebrew/opt/openexr/lib"
    copy_keg_lib libIex-3_4.33.dylib \
        "$BREW_PREFIX/opt/openexr/lib" \
        "/usr/local/opt/openexr/lib" \
        "/opt/homebrew/opt/openexr/lib"
    copy_keg_lib libIlmThread-3_4.33.dylib \
        "$BREW_PREFIX/opt/openexr/lib" \
        "/usr/local/opt/openexr/lib" \
        "/opt/homebrew/opt/openexr/lib"
    copy_keg_lib libImath-3_2.30.dylib \
        "$BREW_PREFIX/opt/imath/lib" \
        "/usr/local/opt/imath/lib" \
        "/opt/homebrew/opt/imath/lib"
    if [ ! -f "$FW_DIR/libpng16.16.dylib" ]; then
        for lib_dir in \
            "$BREW_PREFIX/opt/libpng/lib" \
            "/usr/local/opt/libpng/lib" \
            "/opt/homebrew/opt/libpng/lib"
        do
            if [ -f "$lib_dir/libpng16.16.dylib" ]; then
                rm -f "$FW_DIR/libpng16.16.dylib"
                cp "$lib_dir/libpng16.16.dylib" "$FW_DIR/"
                install_name_tool -id "@rpath/libpng16.16.dylib" \
                    "$FW_DIR/libpng16.16.dylib"
                break
            fi
        done
    fi
    if [ ! -f "$FW_DIR/libarchive.13.dylib" ]; then
        for lib_dir in \
            "$BREW_PREFIX/opt/libarchive/lib" \
            "/usr/local/opt/libarchive/lib" \
            "/opt/homebrew/opt/libarchive/lib"
        do
            if [ -f "$lib_dir/libarchive.13.dylib" ]; then
                rm -f "$FW_DIR/libarchive.13.dylib"
                cp "$lib_dir/libarchive.13.dylib" "$FW_DIR/"
                install_name_tool -id "@rpath/libarchive.13.dylib" \
                    "$FW_DIR/libarchive.13.dylib"
                break
            fi
        done
    fi

    copy_missing_deps() {
        local changed=0
        local dep
        local libname
        local candidate
        local search_dirs=("$BREW_PREFIX/lib" "/usr/local/lib" "/opt/homebrew/lib")

        while read -r dep; do
            libname=$(basename "$dep")
            if [ -f "$FW_DIR/$libname" ]; then
                continue
            fi
            candidate=""
            for base in "${search_dirs[@]}"; do
                if [ -f "$base/$libname" ]; then
                    candidate="$base/$libname"
                    break
                fi
            done
            if [ -z "$candidate" ]; then
                for base in "$BREW_PREFIX/opt" "/usr/local/opt" "/opt/homebrew/opt"; do
                    if [ -d "$base" ]; then
                        for opt_lib in "$base"/*/lib; do
                            if [ -f "$opt_lib/$libname" ]; then
                                candidate="$opt_lib/$libname"
                                break
                            fi
                        done
                    fi
                    if [ -n "$candidate" ]; then
                        break
                    fi
                done
            fi
            if [ -n "$candidate" ]; then
                rm -f "$FW_DIR/$libname"
                cp "$candidate" "$FW_DIR/"
                install_name_tool -id "@rpath/$libname" \
                    "$FW_DIR/$libname"
                changed=1
            fi
        done < <(otool -L "$BIN_DIR/Rayforge" "$FW_DIR"/*.dylib 2>/dev/null | \
            awk '{print $1}' | grep -E '^/usr/local/|^/opt/homebrew/' | \
            sort -u || true)

        return $changed
    }

    # Iteratively pull in any Homebrew deps referenced by bundled binaries.
    for _ in 1 2 3; do
        copy_missing_deps || break
    done

    # Fix all library references to use @rpath instead of absolute paths
    echo "Fixing library references..."
    {
        chmod -R u+w "$FW_DIR" "$BIN_DIR" 2>/dev/null || true
        find "$FW_DIR" -name "*.dylib" -print0 | while IFS= read -r -d '' dylib; do
            otool -L "$dylib" | grep -E '/usr/local/|/opt/homebrew/' | \
                awk '{print $1}' | while read dep; do
                libname=$(basename "$dep")
                if [ -f "$FW_DIR/$libname" ]; then
                    install_name_tool -change "$dep" "@rpath/$libname" "$dylib" 2>/dev/null || true
                fi
            done || true
        done
        for bin in "$BIN_DIR/Rayforge" "$BIN_DIR/Rayforge.bin"; do
            [ -f "$bin" ] || continue
            if ! file "$bin" | grep -q "Mach-O"; then
                continue
            fi
            otool -L "$bin" | grep -E '/usr/local/|/opt/homebrew/' | \
                awk '{print $1}' | while read dep; do
                libname=$(basename "$dep")
                if [ -f "$FW_DIR/$libname" ]; then
                    install_name_tool -change "$dep" "@rpath/$libname" "$bin" 2>/dev/null || true
                fi
            done || true
        done

        # Force libpng references to @rpath to avoid runtime lookups in Homebrew.
        for target in "$FW_DIR"/*.dylib "$BIN_DIR/Rayforge.bin"; do
            [ -f "$target" ] || continue
            otool -L "$target" | awk '{print $1}' | \
                grep -E '/opt/homebrew/opt/libpng/|/usr/local/opt/libpng/' | \
                while read dep; do
                    install_name_tool -change "$dep" "@rpath/libpng16.16.dylib" \
                        "$target" 2>/dev/null || true
                done || true
        done
        if [ -f "$FW_DIR/libfreetype.6.dylib" ]; then
            otool -L "$FW_DIR/libfreetype.6.dylib" | awk '{print $1}' | \
                grep -E '/opt/homebrew/opt/libpng/|/usr/local/opt/libpng/' | \
                while read dep; do
                    install_name_tool -change "$dep" "@rpath/libpng16.16.dylib" \
                        "$FW_DIR/libfreetype.6.dylib" 2>/dev/null || true
                done || true
        fi
        if [ -f "$FW_DIR/libfontconfig.1.dylib" ]; then
            if [ -f "$FW_DIR/libfreetype.6.dylib" ]; then
                otool -L "$FW_DIR/libfontconfig.1.dylib" | awk '{print $1}' | \
                    grep -E '/opt/homebrew/opt/freetype/|/usr/local/opt/freetype/' | \
                    while read dep; do
                        install_name_tool -change "$dep" "@rpath/libfreetype.6.dylib" \
                            "$FW_DIR/libfontconfig.1.dylib" 2>/dev/null || true
                    done || true
            fi
            if [ -f "$FW_DIR/libintl.8.dylib" ]; then
                otool -L "$FW_DIR/libfontconfig.1.dylib" | awk '{print $1}' | \
                    grep -E '/opt/homebrew/opt/gettext/|/usr/local/opt/gettext/' | \
                    while read dep; do
                        install_name_tool -change "$dep" "@rpath/libintl.8.dylib" \
                            "$FW_DIR/libfontconfig.1.dylib" 2>/dev/null || true
                    done || true
            fi
        fi
    } || true

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

    # Re-sign after install_name_tool and dylib rewrites to keep
    # macOS code-signing validation valid on Apple Silicon.
    echo "Re-signing app bundle..."
    rm -rf "$APP_ROOT/_CodeSignature"
    find "dist/Rayforge.app" -type f | while read -r file_path; do
        file_type=$(file -b "$file_path" || true)
        if [[ "$file_type" == *"Mach-O"* ]]; then
            codesign --force --sign - "$file_path"
        fi
    done
    codesign --force --deep --sign - "dist/Rayforge.app"
    codesign --verify --deep --strict --verbose=2 "dist/Rayforge.app"

    # TODO: Bundle vips modules and gdk-pixbuf loaders when vips is installed with SVG support
    # if [ -d "/usr/local/lib/vips-modules-8.17" ]; then
    #     cp -r "/usr/local/lib/vips-modules-8.17" "$FW_DIR/" || true
    # fi
    # if [ -d "/usr/local/lib/gdk-pixbuf-2.0" ]; then
    #     cp -r "/usr/local/lib/gdk-pixbuf-2.0" "$FW_DIR/" || true
    # fi

    # Make sure the plist still points to the wrapper.
    /usr/libexec/PlistBuddy -c "Set :CFBundleExecutable Rayforge" \
        "$APP_ROOT/Info.plist" 2>/dev/null || true

    echo "Cleaning dist/*.whl and dist/*.gz after app bundle..."
    rm -f dist/*.whl dist/*.gz dist/*.tar.gz
fi

if (( DO_DMG == 1 )); then
    echo ""
    echo ""
    print_info "  .dmg Distribution package"
    print_info "--------------------------------------"
    echo ""
    echo "Creating DMG..."
    if [ ! -d "dist/Rayforge.app" ]; then
        echo "dist/Rayforge.app not found.\nBuild the app bundle first." >&2
        exit 1
    fi
    DMG_PATH="dist/Rayforge_${VERSION}.dmg"
    rm -f "$DMG_PATH"
    hdiutil create -volname "Rayforge" -srcfolder "dist/Rayforge.app" \
        -ov -format UDZO "$DMG_PATH"
fi

if (( DO_BUILD == 1 )) && (( DO_BUNDLE == 1 )) && (( DO_DMG == 1 )); then
    echo "Build artifacts created in dist/, dist/*.whl, dist/Rayforge.app, and dist/Rayforge.dmg"
elif (( DO_BUILD == 1 )) && (( DO_BUNDLE == 1 )); then
    echo "Build artifacts created in dist/, dist/*.whl, and dist/Rayforge.app"
elif (( DO_BUILD == 1 )); then
    echo "Build artifacts created in dist/ and dist/*.whl"
elif (( DO_BUNDLE == 1 )); then
    echo "App bundle created in dist/Rayforge.app"
elif (( DO_DMG == 1 )); then
    echo "DMG created in dist/Rayforge.dmg"
fi

echo ""
echo ""
print_info "  Finished!"
echo ""
