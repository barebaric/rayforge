#!/usr/bin/env bash
# Temporarily run pixi against a local raydriver checkout, then restore.
#
# Usage:
#   scripts/pixi-raydriver.sh <pixi args...>
#
# Examples:
#   scripts/pixi-raydriver.sh run lint
#   scripts/pixi-raydriver.sh run test
#   scripts/pixi-raydriver.sh shell
#
# This appends a raydriver dependency-override pointing at a local
# checkout to pixi.toml, runs the given pixi command, and restores the
# original pixi.toml and pixi.lock on exit (also on error or Ctrl-C).
# The override replaces raydriver everywhere, including transitive
# requirements (e.g. rayforge's raydriver pin).
#
# The local checkout defaults to external/raydriver; override with
# RAYDRIVER_PATH. The path is canonicalized because pixi canonicalizes
# symlink paths inconsistently in its lock staleness check, which would
# otherwise make it re-solve the environment on every command.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIXI_TOML="$ROOT_DIR/pixi.toml"
PIXI_LOCK="$ROOT_DIR/pixi.lock"

RAYDRIVER_PATH="${RAYDRIVER_PATH:-$ROOT_DIR/external/raydriver}"
if [[ ! -d "$RAYDRIVER_PATH" ]]; then
    echo "pixi-raydriver: local raydriver checkout not found at '$RAYDRIVER_PATH'." >&2
    echo "             create external/raydriver or set RAYDRIVER_PATH." >&2
    exit 1
fi
RAYDRIVER_ABS="$(cd "$RAYDRIVER_PATH" && pwd)"

if [[ ! -f "$PIXI_TOML" ]]; then
    echo "pixi-raydriver: pixi.toml not found at '$PIXI_TOML'." >&2
    exit 1
fi

MARKER="# pixi-raydriver: temporary override (auto-removed)"
if grep -qF "$MARKER" "$PIXI_TOML"; then
    echo "pixi-raydriver: $PIXI_TOML already has a temporary override marker." >&2
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
raydriver = { path = "$RAYDRIVER_ABS", editable = true }
EOF

cd "$ROOT_DIR"
echo "pixi-raydriver: using local raydriver from $RAYDRIVER_ABS" >&2

# Run pixi without set -e so we can restore and still preserve its exit code.
set +e
pixi "$@"
rc=$?
set -e
exit $rc
