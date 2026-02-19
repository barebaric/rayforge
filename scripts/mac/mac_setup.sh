#!/usr/bin/env bash
set -euo pipefail

INSTALL=0
RUN_APP=0
ENV_WRITTEN=0
SKIP_DEPS=()
GREEN="\033[0;32m"
NC="\033[0m"

print_info() {
    local title=$1
    printf "${GREEN}%s${NC}\n" "$title"
}

should_skip_dep() {
    local dep="$1"
    local skip
    for skip in "${SKIP_DEPS[@]}"; do
        if [[ "$skip" == "$dep" ]]; then
            return 0
        fi
    done
    return 1
}

for arg in "$@"; do
    case "$arg" in
        --install)
            INSTALL=1
            ;;
        --skip-*)
            SKIP_DEPS+=("${arg#--skip-}")
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

if (( INSTALL == 1 )); then
    INSTALL=1
else
    echo ""
    print_info "======================================"
    print_info "    Rayforge macOS Setup Script"
    print_info "======================================"
    echo ""
    echo "Select setup option:"
    echo "    1) Check dependencies only"
    echo "    2) Install missing dependencies"
    echo "    3) Run Rayforge"
    echo "    4) Exit"
    echo ""
    echo "  Tip: You can skip Homebrew checks for specific packages."
    echo "  Example: ./scripts/mac/mac_setup.sh --skip-libvips --skip-openslide"
    echo ""
    read -r -p "Choice (1-4): " SETUP_CHOICE
    case "$SETUP_CHOICE" in
        1)
            INSTALL=0
            ;;
        2)
            INSTALL=1
            ;;
        3)
            RUN_APP=1
            ;;
        4)
            exit 0
            ;;
        *)
            exit 0
            ;;
    esac
fi

if (( RUN_APP == 0 )); then
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
        if should_skip_dep "$dep"; then
            continue
        fi
        if ! brew list --versions "$dep" >/dev/null 2>&1; then
            MISSING+=("$dep")
        fi
    done

    if (( ${#MISSING[@]} > 0 )); then
        if (( INSTALL == 1 )); then
            brew install "${MISSING[@]}"
        else
            echo "Missing Homebrew packages: ${MISSING[*]}" >&2
            echo "Run again and choose 'Install missing dependencies'." >&2
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
    ENV_WRITTEN=1
fi

if (( RUN_APP == 1 )); then
    VENV_PATH=${VENV_PATH:-.venv}
    if [ ! -d "$VENV_PATH" ]; then
        python3 -m venv "$VENV_PATH"
    fi
    # shellcheck source=/dev/null
    source "$VENV_PATH/bin/activate"
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .
    if [ -f .mac_env ]; then
        source .mac_env
    fi
    python -m rayforge.app --loglevel=DEBUG
fi

echo ""
if (( ENV_WRITTEN == 1 )); then
    echo "Environment written to .mac_env"
    echo "Source it before building: source .mac_env"
else
    echo "Environment file not updated in this run."
fi
echo ""
echo ""
print_info "  Finished!"
echo ""
echo ""
echo "To run Rayforge after setup:"
echo "  source .mac_env"
echo "  python3 -m venv .venv"
echo "  source .venv/bin/activate"
echo "  python -m pip install --upgrade pip"
echo "  python -m pip install -r requirements.txt"
echo "  python -m pip install -e ."
echo "  python -m rayforge.app --loglevel=DEBUG"
echo ""
echo ""
