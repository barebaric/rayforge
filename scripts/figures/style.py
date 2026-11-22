"""Palette and fonts shared by the website figure generators."""

from pathlib import Path

from PIL import ImageFont

FONT_DIRS = [
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/liberation"),
    Path("/usr/share/fonts/truetype/freefont"),
]

BG = "white"
CARD_FILL = "#f7f8fa"
CARD_BORDER = "#e3e6eb"
LINE = "#2c313a"
GUIDE = "#c3c9d2"
MUTED = "#6b7280"
TRAVEL = "#e07a30"
START = "#2f6fd0"


def find_font(name: str, size: int) -> ImageFont.FreeTypeFont | None:
    for directory in FONT_DIRS:
        path = directory / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                return None
    return None
