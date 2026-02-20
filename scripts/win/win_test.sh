#!/bin/bash
# This script runs the test suite in the configured MSYS2 environment.
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

echo "--- Running Backend Tests ---"
$PYTHON_EXEC -m pytest -v -ra -m "not ui" --log-cli-level=DEBUG --log-cli-format='%(asctime)s [%(levelname)8s] %(message)s' --log-cli-date-format='%Y-%m-%d %H:%M:%S'

echo "--- Running Stress Tests ---"
$PYTHON_EXEC -m pytest -v -ra -m "stress" --log-cli-level=DEBUG --log-cli-format='%(asctime)s [%(levelname)8s] %(message)s' --log-cli-date-format='%Y-%m-%d %H:%M:%S'

echo "--- Running UI Tests ---"
$PYTHON_EXEC -m pytest -v -ra -m "ui" --log-cli-level=DEBUG --log-cli-format='%(asctime)s [%(levelname)8s] %(message)s' --log-cli-date-format='%Y-%m-%d %H:%M:%S'

echo "✅ All tests passed."
