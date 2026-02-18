#!/bin/bash
set -e

SVG_PATH="website/content/assets/favicon.png"
ICO_PATH="rayforge.ico"

if [ ! -f "$SVG_PATH" ]; then
    echo "FATAL: Icon source file not found at $SVG_PATH"
    exit 1
fi

echo "--- Generating Windows Icon ($ICO_PATH) from SVG ---"

# Use ImageMagick to render the SVG to multiple PNG sizes
# and then combine them into a single .ico file.
# Using a high density ensures the initial render from SVG is high quality.
magick -density 300 -background transparent "$SVG_PATH" \
       -define icon:auto-resize=256,64,48,32,16 \
       "$ICO_PATH"

echo "✅ Icon generation complete."
