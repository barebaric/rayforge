#!/usr/bin/env bash
# Safely generate an image with a local diffusion model.
#
# This wraps scripts/media/generate_image.py in a systemd-run cgroup that caps
# the job's memory and swap, so if the model load spikes it can only kill this
# job -- never other applications (previous runs without the cage caused the
# Linux OOM killer to take down the desktop).
#
# If a virtualenv is active ($VIRTUAL_ENV is set), its python is used;
# otherwise whatever `python3` is on PATH.  (systemd-run doesn't inherit the
# shell's PATH, so we resolve the python path before launching.)
#
# Usage:
#   scripts/media/gen_image.sh --prompt "a wooden CNC part" --out part.png
#
# All arguments are passed through to generate_image.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Memory cap: 36 GB RAM, no swap. OOMPolicy=kill confines any OOM kill to this
# job's cgroup, protecting the rest of the desktop.
# Resolve python3: prefer the venv's python if a .venv exists in the caller's
# working directory, otherwise fall back to whatever python3 is on PATH.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    PYTHON3="$VIRTUAL_ENV/bin/python3"
else
    PYTHON3="$(command -v python3)"
fi

PYTORCH_ALLOC_CONF="expandable_segments:True" \
systemd-run --user --scope -q \
    -p MemoryMax=36G \
    -p MemorySwapMax=0 \
    -p OOMPolicy=kill \
    "$PYTHON3" "$SCRIPT_DIR/generate_image.py" "$@"
