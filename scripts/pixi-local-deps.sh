#!/usr/bin/env bash
# Temporarily run pixi against local raygeo and raydriver checkouts,
# then restore.
#
# Usage:
#   scripts/pixi-local-deps.sh <pixi args...>
#
# Examples:
#   scripts/pixi-local-deps.sh run lint
#   scripts/pixi-local-deps.sh run test
#
# This appends dependency-overrides pointing at local checkouts to
# pixi.toml, runs the given pixi command, and restores the original
# pixi.toml and pixi.lock on exit (also on error or Ctrl-C). The
# overrides replace raygeo and raydriver everywhere, including
# transitive requirements.
#
# This is the combined form of scripts/pixi-raygeo.sh and
# scripts/pixi-raydriver.sh for changes that span both libraries.
#
# The checkouts default to external/raygeo and external/raydriver;
# override with RAYGEO_PATH and RAYDRIVER_PATH. Paths are canonicalized
# because pixi canonicalizes symlink paths inconsistently in its lock
# staleness check.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIXI_TOML="$ROOT_DIR/pixi.toml"
PIXI_LOCK="$ROOT_DIR/pixi.lock"

RAYGEO_PATH="${RAYGEO_PATH:-$ROOT_DIR/external/raygeo}"
RAYDRIVER_PATH="${RAYDRIVER_PATH:-$ROOT_DIR/external/raydriver}"
for name in RAYGEO_PATH RAYDRIVER_PATH; do
    path="${!name}"
    if [[ ! -d "$path" ]]; then
        echo "pixi-local-deps: local checkout not found at '$path'." >&2
        echo "                 set $name or link it under external/." >&2
        exit 1
    fi
done
RAYGEO_ABS="$(cd "$RAYGEO_PATH" && pwd)"
RAYDRIVER_ABS="$(cd "$RAYDRIVER_PATH" && pwd)"

if [[ ! -f "$PIXI_TOML" ]]; then
    echo "pixi-local-deps: pixi.toml not found at '$PIXI_TOML'." >&2
    exit 1
fi

MARKER="# pixi-local-deps: temporary override (auto-removed)"
if grep -qF "$MARKER" "$PIXI_TOML"; then
    echo "pixi-local-deps: $PIXI_TOML already has a temporary override marker." >&2
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
raydriver = { path = "$RAYDRIVER_ABS", editable = true }
EOF

cd "$ROOT_DIR"
echo "pixi-local-deps: using raygeo from $RAYGEO_ABS" >&2
echo "pixi-local-deps: using raydriver from $RAYDRIVER_ABS" >&2

set +e
pixi "$@"
rc=$?
set -e
exit $rc
