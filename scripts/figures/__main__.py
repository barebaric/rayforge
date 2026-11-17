"""Regenerate the website documentation figures.

Usage::

    python -m scripts.figures            # all figures
    python -m scripts.figures bidir_offset invert_example
"""

import argparse

from PIL import Image

from . import bidir_offset, invert_example, path_optimization
from .canvas import save_webp

FIGURES = {
    "bidir_offset": bidir_offset,
    "invert_example": invert_example,
    "path_optimization": path_optimization,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate figures for the website documentation."
    )
    parser.add_argument(
        "names",
        nargs="*",
        choices=[*FIGURES.keys(), "all"],
        help="Figures to generate; defaults to all",
    )
    args = parser.parse_args()
    names = args.names or ["all"]
    if "all" in names:
        names = list(FIGURES.keys())
    for name in names:
        images: list[tuple[str, Image.Image]] = FIGURES[name].generate()
        for filename, image in images:
            save_webp(filename, image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
