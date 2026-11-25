#!/usr/bin/env bash
# Temporarily run pixi against local raygeo and raydriver checkouts.
#
# Usage:
#   scripts/pixi-local.sh <pixi args...>
#
# Examples:
#   scripts/pixi-local.sh run lint
#   scripts/pixi-local.sh run test
#   scripts/pixi-local.sh shell
#
# This appends dependency-overrides pointing at local checkouts to
# pixi.toml, runs the given pixi command, and restores the original
# pixi.toml and pixi.lock on exit (also on error or Ctrl-C). The
# overrides replace the packages everywhere, including transitive
# requirements (e.g. rayforge's own pins).
#
# Either package can be overridden alone: a missing checkout is skipped
# with a warning. The checkouts default to external/raygeo and
# external/raydriver; override with RAYGEO_PATH and RAYDRIVER_PATH.
# Paths are canonicalized because pixi canonicalizes symlink paths
# inconsistently in its lock staleness check, which would otherwise
# make it re-solve the environment on every command.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIXI_TOML="$ROOT_DIR/pixi.toml"
PIXI_LOCK="$ROOT_DIR/pixi.lock"

if [[ ! -f "$PIXI_TOML" ]]; then
    echo "pixi-local: pixi.toml not found at '$PIXI_TOML'." >&2
    exit 1
fi

MARKER="# pixi-local: temporary overrides (auto-removed)"
if grep -qF "$MARKER" "$PIXI_TOML"; then
    echo "pixi-local: $PIXI_TOML already has a temporary override marker." >&2
    echo "           a previous run may not have restored it; run:" >&2
    echo "           git checkout pixi.toml pixi.lock" >&2
    exit 1
fi

declare -a override_lines=()

add_override() {
    local name="$1"
    local path="$2"
    if [[ -z "$path" ]]; then
        return
    fi
    if [[ ! -d "$path" ]]; then
        echo "pixi-local: local $name checkout not found at '$path'." >&2
        echo "           create external/$name or set ${name^^}_PATH." >&2
        exit 1
    fi
    local abs
    abs="$(cd "$path" && pwd)"
    override_lines+=("$name = { path = \"$abs\", editable = true }")
    echo "pixi-local: using local $name from $abs" >&2
}

RAYGEO_PATH="${RAYGEO_PATH:-$ROOT_DIR/external/raygeo}"
RAYDRIVER_PATH="${RAYDRIVER_PATH:-$ROOT_DIR/external/raydriver}"

if [[ -d "$RAYGEO_PATH" ]]; then
    add_override raygeo "$RAYGEO_PATH"
elif [[ -n "${RAYGEO_PATH:-}" && "$RAYGEO_PATH" != "$ROOT_DIR/external/raygeo" ]]; then
    add_override raygeo "$RAYGEO_PATH"
fi

if [[ -d "$RAYDRIVER_PATH" ]]; then
    add_override raydriver "$RAYDRIVER_PATH"
elif [[ -n "${RAYDRIVER_PATH:-}" && "$RAYDRIVER_PATH" != "$ROOT_DIR/external/raydriver" ]]; then
    add_override raydriver "$RAYDRIVER_PATH"
fi

if [[ ${#override_lines[@]} -eq 0 ]]; then
    echo "pixi-local: no local checkouts to override." >&2
    echo "           create external/raygeo or external/raydriver," >&2
    echo "           or set RAYGEO_PATH and/or RAYDRIVER_PATH." >&2
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

{
    echo ""
    echo "$MARKER"
    echo "[pypi-options.dependency-overrides]"
    printf '%s\n' "${override_lines[@]}"
} >> "$PIXI_TOML"

cd "$ROOT_DIR"

# Run pixi without set -e so we can restore and still preserve its exit code.
set +e
pixi "$@"
rc=$?
set -e
exit $rc
