#!/usr/bin/env bash
set -euo pipefail

INSTALL=0
if [[ "${1:-}" == "--install" ]]; then
    INSTALL=1
fi

if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required to set up the macOS toolchain." >&2
    exit 1
fi

BREW_PREFIX=$(brew --prefix)
LIBFFI_PREFIX=$(brew --prefix libffi 2>/dev/null || true)
if [[ -z "$LIBFFI_PREFIX" ]]; then
    LIBFFI_PREFIX="$BREW_PREFIX/opt/libffi"
fi

DEPS=(
    gtk4
    libadwaita
    gobject-introspection
    librsvg
    libvips
    openslide
    pkg-config
    meson
    ninja
    cairo
    pango
    harfbuzz
)

MISSING=()
for dep in "${DEPS[@]}"; do
    if ! brew list --versions "$dep" >/dev/null 2>&1; then
        MISSING+=("$dep")
    fi
done

if (( ${#MISSING[@]} > 0 )); then
    if (( INSTALL == 1 )); then
        brew install "${MISSING[@]}"
    else
        echo "Missing Homebrew packages: ${MISSING[*]}" >&2
        echo "Re-run with --install to install them automatically." >&2
        exit 1
    fi
fi

cat > .mac_env <<EOF
export BREW_PREFIX="$BREW_PREFIX"
export PATH="$BREW_PREFIX/bin:\$PATH"
export PKG_CONFIG_PATH="$BREW_PREFIX/lib/pkgconfig:$BREW_PREFIX/share/pkgconfig:$LIBFFI_PREFIX/lib/pkgconfig:\${PKG_CONFIG_PATH:-}"
export GI_TYPELIB_PATH="$BREW_PREFIX/lib/girepository-1.0:\${GI_TYPELIB_PATH:-}"
export DYLD_FALLBACK_LIBRARY_PATH="$BREW_PREFIX/lib:\${DYLD_FALLBACK_LIBRARY_PATH:-}"
if ! echo "\${PKG_CONFIG_PATH:-}" | tr ':' '\n' | grep -qx "/usr/local/lib/pkgconfig"; then
  export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:\$PKG_CONFIG_PATH"
fi
EOF

echo "Environment written to .mac_env"
echo "Source it before building: source .mac_env"
