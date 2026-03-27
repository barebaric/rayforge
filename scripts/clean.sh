#!/bin/bash
find . -type d \( -name __pycache__ -o -name "*.egg-info" \) -exec rm -r {} + 2>/dev/null || true
find . -type f \( -name "*.mo" -o -name "rayforge.po~" \) -delete
