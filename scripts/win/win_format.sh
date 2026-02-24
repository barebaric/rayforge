#!/bin/bash
# This script runs the auto-formatter in the configured MSYS2 environment.
set -e

# Ensure the MSYS2 environment is configured.
if [ ! -f .msys2_env ]; then
    echo "FATAL: .msys2_env file not found. Please run 'bash scripts/win/win_setup.sh' first."
    exit 1
fi

# Load the environment variables (MSYS2_PATH, PKG_CONFIG_PATH, etc.)
source .msys2_env

# Define the Python executable for convenience
PYTHON_EXEC="$MSYS2_PATH/mingw64/bin/python"

echo "--- Running ruff format ---"
$PYTHON_EXEC -m ruff format rayforge tests scripts "$@"

echo "--- Running ruff auto-fix ---"
$PYTHON_EXEC -m ruff check --fix rayforge tests scripts "$@"

echo "✅ Formatting complete."
