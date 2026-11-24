#!/usr/bin/env python3
"""Generate the PNG assets referenced by the MSIX manifest."""

import argparse
import sys
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SOURCE = _ROOT / "website/static/images/favicon.png"
_ASSETS = {
    "StoreLogo.png": (50, 50),
    "Square44x44Logo.png": (44, 44),
    "Square150x150Logo.png": (150, 150),
    "FileIcon.png": (44, 44),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the PNG assets referenced by the MSIX manifest."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=_DEFAULT_SOURCE,
        help="Source image used to generate the assets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where the generated assets are written.",
    )
    args = parser.parse_args(argv)

    image = Image.open(args.source)
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    args.output.mkdir(parents=True, exist_ok=True)
    for name, size in _ASSETS.items():
        resized = image.resize(size, Image.Resampling.LANCZOS)
        resized.save(args.output / name, format="PNG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
