#!/usr/bin/env bash
# Rebuild raygeo from external/raygeo after Rust/Python source changes.
# Clears the uv wheel cache so the new .so is compiled on reinstall.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIXI_TOML="$ROOT_DIR/pixi.toml"
PIXI_LOCK="$ROOT_DIR/pixi.lock"

RAYGEO_CANDIDATE="${RAYGEO_PATH:-$ROOT_DIR/external/raygeo}"
if [[ ! -d "$RAYGEO_CANDIDATE" ]]; then
    echo "rebuild-raygeo: local raygeo checkout not found at '$RAYGEO_CANDIDATE'." >&2
    echo "                create external/raygeo or set RAYGEO_PATH." >&2
    exit 1
fi
RAYGEO_ABS="$(cd "$RAYGEO_CANDIDATE" && pwd)"

# Clear the uv cache so the next build is fresh, not a stale wheel.
CACHE_DIR=$(pixi info --json | python3 -c 'import json,sys;print(json.load(sys.stdin)["cache_dir"]+"/uv-cache")')
UV_CACHE_DIR="$CACHE_DIR" uv cache prune

MARKER="# pixi-raygeo: temporary override (auto-removed)"
if grep -qF "$MARKER" "$PIXI_TOML"; then
    echo "rebuild-raygeo: $PIXI_TOML already has a temporary override marker." >&2
    echo "                a previous run may not have restored it; run:" >&2
    echo "                git checkout pixi.toml pixi.lock" >&2
    exit 1
fi

backup="$(mktemp -d)"
cleanup() {
    cp "$backup/pixi.toml" "$PIXI_TOML"
    if [[ -f "$backup/pixi.lock" ]]; then
        cp "$backup/pixi.lock" "$PIXI_LOCK"
    fi
    rm -rf "$backup"
}
trap cleanup EXIT

cp "$PIXI_TOML" "$backup/pixi.toml"
if [[ -f "$PIXI_LOCK" ]]; then
    cp "$PIXI_LOCK" "$backup/pixi.lock"
fi

cat >> "$PIXI_TOML" <<EOF

$MARKER
[pypi-options.dependency-overrides]
raygeo = { path = "$RAYGEO_ABS", editable = true }
EOF

cd "$ROOT_DIR"

# First solve so the lock records the local checkout, then force a rebuild.
pixi install
pixi reinstall raygeo
