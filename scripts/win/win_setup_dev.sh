#!/bin/bash
# This script installs development tools (linters, formatters) in the configured MSYS2 environment.
set -e

# Ensure the MSYS2 environment is configured.
if [ ! -f .msys2_env ]; then
    echo "FATAL: .msys2_env file not found. Please run 'bash scripts/win/win_setup.sh' first."
    exit 1
fi

source .msys2_env

PYTHON_BIN_PATH="$MSYS2_PATH/mingw64/bin/python"

echo "--- Installing Development Tools ---"

$PYTHON_BIN_PATH -m pip install --no-cache-dir flake8 pyflakes pyright ruff pre-commit --break-system-packages

echo "--- Installing Git Pre-Commit Hooks ---"
$PYTHON_BIN_PATH -m pre_commit install

echo "✅ Development tools installed."
