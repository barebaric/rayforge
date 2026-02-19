#!/bin/bash
# Script to generate macOS ICNS icon from SVG source
# Uses only native macOS tools: rsvg-convert and iconutil
#
# Requirements:
#   - rsvg-convert: brew install librsvg
#   - iconutil: built-in on macOS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Paths
SVG_PATH="$PROJECT_ROOT/website/static/assets/icon-app.svg"
ICONSET_PATH="$PROJECT_ROOT/build/icon.iconset"
OUTPUT_PATH="$PROJECT_ROOT/rayforge.icns"

# Check if SVG exists
if [ ! -f "$SVG_PATH" ]; then
    echo -e "${RED}Error: SVG file not found at $SVG_PATH${NC}"
    exit 1
fi

# Check if rsvg-convert is installed
if ! command -v rsvg-convert &> /dev/null; then
    echo -e "${RED}Error: rsvg-convert not found${NC}"
    echo "Install with: brew install librsvg"
    exit 1
fi

# Check if iconutil is available (should be on all macOS systems)
if ! command -v iconutil &> /dev/null; then
    echo -e "${RED}Error: iconutil not found. This script requires macOS.${NC}"
    exit 1
fi

echo -e "${GREEN}Source SVG:${NC} $SVG_PATH"
echo -e "${GREEN}Output ICNS:${NC} $OUTPUT_PATH"
echo ""

# Create build directory
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Remove existing iconset if it exists
if [ -d "$ICONSET_PATH" ]; then
    echo -e "${YELLOW}Removing existing iconset...${NC}"
    rm -rf "$ICONSET_PATH"
fi

# Create iconset directory
echo -e "${GREEN}Creating iconset directory...${NC}"
mkdir -p "$ICONSET_PATH"

# Function to generate PNG at specific size
generate_png() {
    local size=$1
    local scale=$2
    local pixel_size=$((size * scale))

    if [ $scale -eq 1 ]; then
        local filename="icon_${size}x${size}.png"
    else
        local filename="icon_${size}x${size}@${scale}x.png"
    fi

    local output_file="$ICONSET_PATH/$filename"

    echo "  Generating ${pixel_size}x${pixel_size} → $filename"
    rsvg-convert -w $pixel_size -h $pixel_size "$SVG_PATH" -o "$output_file"
}

# Generate all required sizes for macOS ICNS
# Format: size scale
echo -e "\n${GREEN}Generating PNG files...${NC}"

# 16x16
generate_png 16 1
generate_png 16 2

# 32x32
generate_png 32 1
generate_png 32 2

# 128x128
generate_png 128 1
generate_png 128 2

# 256x256
generate_png 256 1
generate_png 256 2

# 512x512
generate_png 512 1
generate_png 512 2

# 1024x1024 (only @2x for 512pt displays)
echo "  Generating 1024x1024 → icon_512x512@2x.png"
rsvg-convert -w 1024 -h 1024 "$SVG_PATH" -o "$ICONSET_PATH/icon_512x512@2x.png"

# Generate ICNS file using iconutil
echo -e "\n${GREEN}Generating ICNS file...${NC}"
iconutil -c icns -o "$OUTPUT_PATH" "$ICONSET_PATH"

# Clean up iconset directory
echo -e "\n${GREEN}Cleaning up temporary files...${NC}"
rm -rf "$ICONSET_PATH"

echo -e "\n${GREEN}✓ Done!${NC} ICNS file created at: ${YELLOW}$OUTPUT_PATH${NC}"
echo -e "File size: $(du -h "$OUTPUT_PATH" | cut -f1)"
