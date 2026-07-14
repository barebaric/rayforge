#!/usr/bin/env bash
# Rebuild raygeo from external/raygeo after Rust/Python source changes.
# Clears the uv wheel cache so the new .so is compiled on reinstall.
set -euo pipefail

CACHE_DIR=$(pixi info --json | python3 -c 'import json,sys;print(json.load(sys.stdin)["cache_dir"]+"/uv-cache")')
UV_CACHE_DIR="$CACHE_DIR" uv cache prune
pixi reinstall -e local-raygeo raygeo