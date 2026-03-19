#!/bin/bash
set -e

SOURCE_PATH="website/static/images/favicon.png"
ICO_PATH="rayforge.ico"

if [ ! -f "$SOURCE_PATH" ]; then
    echo "FATAL: Icon source file not found at $SOURCE_PATH"
    exit 1
fi

echo "--- Generating Windows Icon ($ICO_PATH) ---"

# Debug: show original paths and working directory
echo "DEBUG: PWD=$PWD"
echo "DEBUG: SOURCE_PATH=$SOURCE_PATH"
echo "DEBUG: ICO_PATH=$ICO_PATH"

# Convert to Windows-style paths with forward slashes
WIN_SOURCE_PATH=$(cygpath -m "$SOURCE_PATH")
WIN_ICO_PATH=$(cygpath -m "$ICO_PATH")

echo "DEBUG: WIN_SOURCE_PATH=$WIN_SOURCE_PATH"
echo "DEBUG: WIN_ICO_PATH=$WIN_ICO_PATH"

# Disable MSYS2's automatic path conversion for magick.exe arguments
MSYS_NO_PATHCONV=1 magick -density 300 -background transparent "$WIN_SOURCE_PATH" \
       -define icon:auto-resize=256,64,48,32,16 \
       "$WIN_ICO_PATH"

echo "✅ Icon generation complete."
