#!/usr/bin/env bash
set -euo pipefail

# Builds a self-contained Linux AppImage of Rayforge.
#
# The AppImage bundles the conda-forge runtime of the pixi "appimage"
# environment (GTK4, libadwaita, Python, and the native imaging stack)
# together with a freshly built rayforge wheel. A custom AppRun wires up
# library paths at launch and generates the pixbuf and fontconfig caches
# there, because the AppImage mount point changes on every start.
#
# The wheel build uses PEP 517 build isolation, so network access to
# PyPI is required.

ARCH=$(uname -m)
ENV_NAME="appimage"
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

if [[ -n "${GITHUB_REF_NAME:-}" && "${GITHUB_REF_TYPE:-}" == "tag" ]]; then
    VERSION="$GITHUB_REF_NAME"
else
    VERSION=$(git describe --tags --always)
fi
VERSION="${VERSION#v}"
echo "Building Rayforge AppImage for version ${VERSION}"

ENV_PREFIX=".pixi/envs/${ENV_NAME}"
if [ ! -d "$ENV_PREFIX" ]; then
    echo "Installing pixi environment '${ENV_NAME}'..."
    pixi install --frozen --environment "$ENV_NAME"
fi

BUILD_DIR=$(mktemp -d)
cleanup() {
    rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

echo "Building wheel..."
"$ENV_PREFIX/bin/python" -m build --wheel --outdir "$BUILD_DIR/wheel"
WHEEL=$(ls "$BUILD_DIR"/wheel/rayforge-*.whl)

APPDIR="$BUILD_DIR/Rayforge.AppDir"
mkdir -p "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/metainfo" \
         "$APPDIR/usr/share/icons/hicolor/scalable/apps"

echo "Copying runtime environment..."
cp -a "$ENV_PREFIX" "$APPDIR/env"

echo "Pruning files that are not needed at runtime..."
rm -rf "$APPDIR/env/share/doc" \
       "$APPDIR/env/share/man" \
       "$APPDIR/env/share/gir-1.0"

echo "Pruning development toolchain..."
"$APPDIR/env/bin/python" - "$APPDIR/env" <<'PY'
import json
import pathlib
import sys

env = pathlib.Path(sys.argv[1])
dev_prefixes = (
    "rust", "cargo", "gcc_impl", "gcc_linux-64", "gxx_impl",
    "gxx_linux-64", "gfortran_impl", "gfortran_linux-64",
    "libgcc-devel", "libstdcxx-devel", "binutils_impl",
    "binutils_linux-64", "ld_impl", "sysroot_linux-64",
    "kernel-headers_linux-64", "patchelf",
)
removed = []
for record in sorted((env / "conda-meta").glob("*.json")):
    info = json.loads(record.read_text())
    if not info["name"].startswith(dev_prefixes):
        continue
    for rel_path in info.get("files", []):
        path = env / rel_path
        if path.is_symlink() or path.is_file():
            path.unlink()
    record.unlink()
    removed.append(info["name"])
print(f"Removed {len(removed)} development packages: {', '.join(removed)}")
PY

SITE_PACKAGES=$("$APPDIR/env/bin/python" -c \
    "import sysconfig; print(sysconfig.get_paths()['purelib'])")

echo "Installing wheel into bundle..."
rm -rf "$SITE_PACKAGES"/__editable__*rayforge* "$SITE_PACKAGES"/rayforge-*.dist-info
"$APPDIR/env/bin/python" -m ensurepip --upgrade
"$APPDIR/env/bin/python" -m pip install --no-index --no-deps "$WHEEL"
"$APPDIR/env/bin/python" -m pip uninstall -y pip
rm -f "$APPDIR/env/bin/rayforge"

echo "Assembling AppDir..."
cp data/org.rayforge.rayforge.desktop "$APPDIR/usr/share/applications/"
cp data/org.rayforge.rayforge.desktop "$APPDIR/"
cp data/org.rayforge.rayforge.metainfo.xml "$APPDIR/usr/share/metainfo/"
ICON="rayforge/resources/icons/org.rayforge.rayforge.svg"
cp "$ICON" "$APPDIR/usr/share/icons/hicolor/scalable/apps/"
cp "$ICON" "$APPDIR/org.rayforge.rayforge.svg"
ln -sf org.rayforge.rayforge.svg "$APPDIR/.DirIcon"

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/bash
set -e

APPDIR="$(dirname "$(readlink -f "$0")")"
BUNDLE="$APPDIR/env"

unset PYTHONHOME
unset PYTHONPATH

export LD_LIBRARY_PATH="$BUNDLE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GI_TYPELIB_PATH="$BUNDLE/lib/girepository-1.0"
export XDG_DATA_DIRS="$BUNDLE/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"
export LOCPATH="$BUNDLE/share/locale${LOCPATH:+:$LOCPATH}"

if [ -f "$BUNDLE/etc/ssl/cacert.pem" ]; then
    export SSL_CERT_FILE="$BUNDLE/etc/ssl/cacert.pem"
fi

if [ -d "$BUNDLE/lib/gio/modules" ]; then
    export GIO_EXTRA_MODULES="$BUNDLE/lib/gio/modules"
fi

CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/rayforge-appimage"
if ! mkdir -p "$CACHE_DIR" 2>/dev/null; then
    CACHE_DIR=$(mktemp -d)
fi

LOADERS_DIR="$BUNDLE/lib/gdk-pixbuf-2.0/2.10.0/loaders"
if [ -d "$LOADERS_DIR" ]; then
    PIXBUF_CACHE="$CACHE_DIR/pixbuf-loaders.cache"
    if "$BUNDLE/bin/gdk-pixbuf-query-loaders" \
            "$LOADERS_DIR"/*.so > "$PIXBUF_CACHE" 2>/dev/null; then
        export GDK_PIXBUF_MODULE_FILE="$PIXBUF_CACHE"
    fi
fi

FONTCONFIG_CONF="$CACHE_DIR/fonts.conf"
cat > "$FONTCONFIG_CONF" <<XMLEOF
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">
<fontconfig>
  <dir>$BUNDLE/share/fonts</dir>
  <dir>/usr/share/fonts</dir>
  <dir>/usr/local/share/fonts</dir>
  <dir prefix="xdg">fonts</dir>
  <include ignore_missing="yes">$BUNDLE/etc/fonts/conf.d</include>
  <cachedir>$CACHE_DIR/fontconfig</cachedir>
</fontconfig>
XMLEOF
export FONTCONFIG_FILE="$FONTCONFIG_CONF"

exec "$BUNDLE/bin/python" -P -m rayforge "$@"
EOF
chmod +x "$APPDIR/AppRun"

mkdir -p dist
APPIMAGE_FILE="dist/Rayforge-${VERSION}-${ARCH}.AppImage"

APPIMAGETOOL="${RAYFORGE_APPIMAGETOOL:-$HOME/.cache/rayforge-build/appimagetool-${ARCH}.AppImage}"
if [ ! -f "$APPIMAGETOOL" ]; then
    echo "Downloading appimagetool..."
    mkdir -p "$(dirname "$APPIMAGETOOL")"
    curl -fsSL -o "$APPIMAGETOOL" \
        "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-${ARCH}.AppImage"
fi
chmod +x "$APPIMAGETOOL"

echo "Packing AppImage..."
APPIMAGE_EXTRACT_AND_RUN=1 "$APPIMAGETOOL" --comp xz "$APPDIR" "$APPIMAGE_FILE"

echo "Created: $APPIMAGE_FILE"
