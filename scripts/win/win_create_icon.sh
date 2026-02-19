#!/bin/bash
set -e

SOURCE_PATH="website/static/assets/favicon.png"
ICO_PATH="rayforge.ico"

if [ ! -f "$SOURCE_PATH" ]; then
    echo "FATAL: Icon source file not found at $SOURCE_PATH"
    exit 1
fi

echo "--- Generating Windows Icon ($ICO_PATH) ---"

magick -density 300 -background transparent "$SOURCE_PATH" \
       -define icon:auto-resize=256,64,48,32,16 \
       "$ICO_PATH"

echo "✅ Icon generation complete."
