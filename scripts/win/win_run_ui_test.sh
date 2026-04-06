#!/bin/bash
# Runs a UI smoke test on the built Windows executable.
# Usage: win_run_ui_test.sh <bundle_dir> <executable_name>
#
# This script runs the executable with --uiscript to verify that
# the UI opens and displays correctly.

set -e

BUNDLE_DIR="${1:-dist/rayforge-latest}"
EXECUTABLE_NAME="${2:-rayforge-latest.exe}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UI_TEST_SCRIPT="${SCRIPT_DIR}/win_ui_test.py"
TIMEOUT_SECONDS=60

if [ ! -d "${BUNDLE_DIR}" ]; then
    echo "ERROR: Bundle directory not found: ${BUNDLE_DIR}"
    exit 1
fi

EXECUTABLE_PATH="${BUNDLE_DIR}/${EXECUTABLE_NAME}"
if [ ! -f "${EXECUTABLE_PATH}" ]; then
    echo "ERROR: Executable not found: ${EXECUTABLE_PATH}"
    exit 1
fi

if [ ! -f "${UI_TEST_SCRIPT}" ]; then
    echo "ERROR: UI test script not found: ${UI_TEST_SCRIPT}"
    exit 1
fi

echo "--- Running UI Smoke Test ---"
echo "Bundle: ${BUNDLE_DIR}"
echo "Executable: ${EXECUTABLE_NAME}"
echo "UI Test Script: ${UI_TEST_SCRIPT}"

cd "${BUNDLE_DIR}"

# Run the executable with the UI test script.
# Use a timeout to prevent hanging if something goes wrong.
# The --uiscript flag runs the test after the UI is fully loaded.
if command -v timeout &> /dev/null; then
    timeout "${TIMEOUT_SECONDS}" ./"${EXECUTABLE_NAME}" --uiscript "${UI_TEST_SCRIPT}"
else
    # Fallback for environments without timeout command
    ./"${EXECUTABLE_NAME}" --uiscript "${UI_TEST_SCRIPT}" &
    PID=$!
    sleep "${TIMEOUT_SECONDS}"
    if kill -0 $PID 2>/dev/null; then
        echo "ERROR: Test timed out after ${TIMEOUT_SECONDS} seconds"
        kill $PID 2>/dev/null || true
        exit 1
    fi
    wait $PID
fi

echo "✅ UI smoke test completed successfully"
