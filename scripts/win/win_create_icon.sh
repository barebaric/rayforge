#!/bin/bash
set -e

SOURCE_PATH="website/static/images/favicon.png"
ICO_PATH="rayforge.ico"

if [ ! -f "$SOURCE_PATH" ]; then
    echo "FATAL: Icon source file not found at $SOURCE_PATH"
    exit 1
fi

echo "--- Generating Windows Icon ($ICO_PATH) ---"

# Convert paths to Windows format for ImageMagick (native Windows binary)
WIN_SOURCE_PATH=$(cygpath -w "$SOURCE_PATH")
WIN_ICO_PATH=$(cygpath -w "$ICO_PATH")

magick -density 300 -background transparent "$WIN_SOURCE_PATH" \
       -define icon:auto-resize=256,64,48,32,16 \
       "$WIN_ICO_PATH"

echo "✅ Icon generation complete."
