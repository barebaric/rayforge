#!/usr/bin/env bash
# Wrapper script to run Blender setup script with flatpak

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

flatpak run --filesystem=host --command=blender org.blender.Blender \
    --background \
    --python "$PROJECT_DIR/scripts/media/generate_blender_setup.py" \
    -- "$@"
