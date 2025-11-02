#!/bin/bash
# This script runs the test suite in the configured MSYS2 environment.
set -e

# Ensure the MSYS2 environment is configured.
if [ ! -f .msys2_env ]; then
    echo "FATAL: .msys2_env file not found. Please run 'bash scripts/setup_windows.sh' first."
    exit 1
fi

# Load the environment variables (MSYS2_PATH, PKG_CONFIG_PATH, etc.)
source .msys2_env

echo "--- Running Tests ---"

# Use the Python interpreter from our MSYS2 environment to run pytest.
$MSYS2_PATH/mingw64/bin/python -m pytest -v -ra

echo "✅ All tests passed."
