#!/bin/bash
set -e

SVG_PATH="rayforge/resources/icons/org.rayforge.rayforge.svg"
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
    \( -clone 0 -resize 256x256 \) \
    \( -clone 0 -resize 64x64 \) \
    \( -clone 0 -resize 48x48 \) \
    \( -clone 0 -resize 32x32 \) \
    \( -clone 0 -resize 16x16 \) \
    -delete 0 -alpha on -colors 256 "$ICO_PATH"

echo "✅ Icon generation complete."
