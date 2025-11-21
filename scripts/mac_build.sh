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
    pyinstaller --onedir --windowed \
        --log-level INFO \
        --name "Rayforge" \
        --osx-bundle-identifier "org.rayforge.rayforge" \
        --add-data "rayforge/version.txt:rayforge" \
        --add-data "rayforge/resources:rayforge/resources" \
        --add-data "rayforge/locale:rayforge/locale" \
        --hidden-import "gi._gi_cairo" \
        rayforge/app.py
fi

echo "Build artifacts created in dist/ and dist/*.whl"
