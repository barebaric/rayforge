#!/bin/bash
# Runs Rayforge from source in the configured MSYS2 environment.
# Usage: win_run.sh [args passed to rayforge]
set -e

if [ ! -f .msys2_env ]; then
    echo "FATAL: .msys2_env file not found. Please run 'bash scripts/win/win_setup.sh' first."
    exit 1
fi

source .msys2_env

exec python -m rayforge.app "$@"
