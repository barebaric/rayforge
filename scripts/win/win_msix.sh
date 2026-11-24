#!/bin/bash
# Build an MSIX package from the PyInstaller bundle created by win_build.sh.
set -e

APP_VERSION=${1:-$(git describe --tags --always || echo "v0.0.0-local")}
CLEAN_VERSION="${APP_VERSION#v}"
BUNDLE_NAME="rayforge-v${CLEAN_VERSION}"
BUNDLE_DIR="dist/${BUNDLE_NAME}"

MSIX_VERSION="${MSIX_VERSION:-$APP_VERSION}"
MSIX_IDENTITY_NAME="${MSIX_IDENTITY_NAME:-Rayforge.Dev}"
MSIX_PUBLISHER="${MSIX_PUBLISHER:-CN=Rayforge Dev}"
MSIX_PUBLISHER_DISPLAY_NAME="${MSIX_PUBLISHER_DISPLAY_NAME:-Rayforge}"
MSIX_OUTPUT_DIR="${MSIX_OUTPUT_DIR:-dist}"
STAGE_DIR="${MSIX_OUTPUT_DIR}/msix/${BUNDLE_NAME}"
PACKAGE_PATH="${MSIX_OUTPUT_DIR}/${BUNDLE_NAME}-x64.msix"

# Load the MSYS2 environment if the build was prepared by win_setup.sh.
if [ -f .msys2_env ]; then
    # shellcheck disable=SC1091
    source .msys2_env
fi

if [ ! -d "$BUNDLE_DIR" ]; then
    echo "FATAL: PyInstaller bundle not found at $BUNDLE_DIR."
    echo "Run scripts/win/win_build.sh first."
    exit 1
fi

if [ ! -f rayforge.ico ]; then
    echo "Creating application icon..."
    python3 scripts/win/win_create_icon.py
fi

echo "--- Building MSIX package (version: $MSIX_VERSION) ---"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
cp -a "${BUNDLE_DIR}/." "$STAGE_DIR/"

echo "Generating MSIX assets..."
python3 scripts/win/msix_create_assets.py \
    --output "$STAGE_DIR/Assets"

echo "Rendering MSIX manifest..."
python3 scripts/win/msix/render_manifest.py \
    --template scripts/win/msix/AppxManifest.xml.in \
    --output "$STAGE_DIR/AppxManifest.xml" \
    --version "$MSIX_VERSION" \
    --executable "${BUNDLE_NAME}.exe" \
    --identity-name "$MSIX_IDENTITY_NAME" \
    --publisher "$MSIX_PUBLISHER" \
    --publisher-display-name "$MSIX_PUBLISHER_DISPLAY_NAME"

to_windows_path() {
    case "$1" in
        [A-Za-z]:\\*)
            printf '%s\n' "$1"
            ;;
        *)
            if command -v cygpath >/dev/null 2>&1; then
                cygpath -w "$1"
            else
                printf '%s\n' "$1"
            fi
            ;;
    esac
}

find_sdk_tool() {
    local name="$1"
    local override="$2"
    if [ -n "$override" ]; then
        printf '%s\n' "$override"
        return 0
    fi
    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return 0
    fi
    if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command \
            "(Get-ChildItem 'C:\\Program Files (x86)\\Windows Kits\\10\\bin\\*\\x64\\${name}' | Sort-Object FullName -Descending | Select-Object -First 1).FullName" \
            2>/dev/null | tr -d '\r'
    fi
}

MAKEAPPX_PATH=$(find_sdk_tool "makeappx.exe" "${MAKEAPPX:-}")
if [ -z "$MAKEAPPX_PATH" ]; then
    echo "FATAL: makeappx.exe was not found."
    echo "Install the Windows SDK or set MAKEAPPX to its full path."
    exit 1
fi

echo "Packing MSIX with $MAKEAPPX_PATH..."
MSYS2_ARG_CONV_EXCL='*' "$MAKEAPPX_PATH" pack /o \
    /d "$(to_windows_path "$STAGE_DIR")" \
    /p "$(to_windows_path "$PACKAGE_PATH")"

if [ -n "${MSIX_CERT_PFX:-}" ]; then
    if [ -z "${MSIX_CERT_PASSWORD:-}" ]; then
        echo "FATAL: MSIX_CERT_PASSWORD is required for signing."
        exit 1
    fi

    CERT_PFX="$MSIX_CERT_PFX"
    if command -v cygpath >/dev/null 2>&1; then
        CERT_PFX=$(cygpath -u "$MSIX_CERT_PFX")
    fi
    if [ ! -f "$CERT_PFX" ]; then
        echo "FATAL: Signing certificate not found at $MSIX_CERT_PFX."
        exit 1
    fi

    SIGNTOOL_PATH=$(find_sdk_tool "signtool.exe" "${SIGNTOOL:-}")
    if [ -z "$SIGNTOOL_PATH" ]; then
        echo "FATAL: signtool.exe was not found."
        echo "Install the Windows SDK or set SIGNTOOL to its full path."
        exit 1
    fi

    TIMESTAMP_ARGS=()
    if [ -n "${MSIX_TIMESTAMP_URL:-}" ]; then
        TIMESTAMP_ARGS=(/td SHA256 /tr "$MSIX_TIMESTAMP_URL")
    fi

    echo "Signing MSIX with $SIGNTOOL_PATH..."
    MSYS2_ARG_CONV_EXCL='*' "$SIGNTOOL_PATH" sign /fd SHA256 \
        "${TIMESTAMP_ARGS[@]}" \
        /f "$(to_windows_path "$CERT_PFX")" \
        /p "$MSIX_CERT_PASSWORD" \
        "$(to_windows_path "$PACKAGE_PATH")"
else
    echo "No signing certificate configured; the MSIX is unsigned."
fi

echo "MSIX package created: $PACKAGE_PATH"
echo "  Identity: $MSIX_IDENTITY_NAME"
echo "  Publisher: $MSIX_PUBLISHER"
echo "  Version: $MSIX_VERSION"
