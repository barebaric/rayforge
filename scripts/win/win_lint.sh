#!/bin/bash
# This script runs the linters in the configured MSYS2 environment.
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

echo "--- Running flake8 ---"
$PYTHON_EXEC -m flake8 --ignore=E127,E128,E121,E123,E126,E203,E226,E24,E704,W503,W504 --builtins=_ rayforge tests "$@"

echo "--- Running pyflakes ---"
PYFLAKES_BUILTINS=_ $PYTHON_EXEC -m pyflakes rayforge tests "$@"

echo "--- Running pyright ---"
$PYTHON_EXEC -m pyright "$@"

echo "✅ All linters passed."
