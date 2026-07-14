#!/usr/bin/env bash
# Temporarily run pixi against a local raygeo checkout, then restore.
#
# Usage:
#   scripts/pixi-raygeo.sh <pixi args...>
#
# Examples:
#   scripts/pixi-raygeo.sh run lint
#   scripts/pixi-raygeo.sh run test
#   scripts/pixi-raygeo.sh shell
#
# This appends a raygeo dependency-override pointing at a local checkout to
# pixi.toml, runs the given pixi command, and restores the original pixi.toml
# and pixi.lock on exit (also on error or Ctrl-C). The override replaces raygeo
# everywhere, including transitive requirements (e.g. rayforge's raygeo pin).
#
# The local checkout defaults to external/raygeo; override with RAYGEO_PATH.
# The path is canonicalized because pixi canonicalizes symlink paths
# inconsistently in its lock staleness check, which would otherwise make it
# re-solve the environment on every command.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIXI_TOML="$ROOT_DIR/pixi.toml"
PIXI_LOCK="$ROOT_DIR/pixi.lock"

RAYGEO_PATH="${RAYGEO_PATH:-$ROOT_DIR/external/raygeo}"
if [[ ! -d "$RAYGEO_PATH" ]]; then
    echo "pixi-raygeo: local raygeo checkout not found at '$RAYGEO_PATH'." >&2
    echo "             create external/raygeo or set RAYGEO_PATH." >&2
    exit 1
fi
RAYGEO_ABS="$(cd "$RAYGEO_PATH" && pwd)"

if [[ ! -f "$PIXI_TOML" ]]; then
    echo "pixi-raygeo: pixi.toml not found at '$PIXI_TOML'." >&2
    exit 1
fi

MARKER="# pixi-raygeo: temporary override (auto-removed)"
if grep -qF "$MARKER" "$PIXI_TOML"; then
    echo "pixi-raygeo: $PIXI_TOML already has a temporary override marker." >&2
    echo "             a previous run may not have restored it; run:" >&2
    echo "             git checkout pixi.toml pixi.lock" >&2
    exit 1
fi

backup="$(mktemp -d)"
cleanup() {
    # Restore the originals no matter how we exit.
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
echo "pixi-raygeo: using local raygeo from $RAYGEO_ABS" >&2

# Run pixi without set -e so we can restore and still preserve its exit code.
set +e
pixi "$@"
rc=$?
set -e
exit $rc
