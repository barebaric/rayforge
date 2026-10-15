#!/usr/bin/env python3
"""Generate the bidirectional scan offset diagrams for the website docs.

Renders how right-to-left raster passes misalign by an amount that
grows with engrave speed, and how the machine's speed-dependent offset
table straightens the edges back out, into two images under
``website/static/images/``. The images carry no translatable words;
titles live in the markdown that embeds them.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "website" / "static" / "images"

BG = "white"
CARD_FILL = "#f7f8fa"
CARD_BORDER = "#e3e6eb"
LINE = "#2c313a"
GUIDE = "#c3c9d2"
MUTED = "#6b7280"

SPEEDS = [1000, 4000, 8000]
UNCORRECTED_OFFSETS = [6, 14, 24]
MAX_OFFSET = max(UNCORRECTED_OFFSETS)

CARD_W, CARD_H = 280, 170
CARD_GAP = 30
CARD_RADIUS = 12
RECT_INSET = 16
LINE_PITCH = 8
LINE_THICKNESS = 3

FONT_DIRS = [
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/liberation"),
    Path("/usr/share/fonts/truetype/freefont"),
]


def find_font(name: str, size: int) -> ImageFont.FreeTypeFont | None:
    for directory in FONT_DIRS:
        path = directory / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                return None
    return None


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    x: int,
    y0: int,
    y1: int,
    dash: int = 5,
    gap: int = 4,
) -> None:
    y = y0
    while y < y1:
        draw.line([(x, y), (x, min(y + dash, y1))], fill=GUIDE, width=1)
        y += dash + gap


def draw_engraved_rect(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    offset: int,
    corrected: bool,
) -> None:
    draw_dashed_line(draw, x - 3, y - 6, y + h + 6)
    draw_dashed_line(draw, x + w + 2, y - 6, y + h + 6)
    for index, line_y in enumerate(range(y, y + h, LINE_PITCH)):
        shift = 0 if corrected or index % 2 == 0 else offset
        draw.rectangle(
            [x + shift, line_y, x + shift + w, line_y + LINE_THICKNESS],
            fill=LINE,
        )


def render(corrected: bool) -> Image.Image:
    label_font = find_font("DejaVuSans.ttf", 18)

    columns = len(SPEEDS)
    width = 24 + columns * CARD_W + (columns - 1) * CARD_GAP + 24
    height = 24 + CARD_H + 12 + 30
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    for col in range(columns):
        cx = 24 + col * (CARD_W + CARD_GAP)
        draw.rounded_rectangle(
            [cx, 24, cx + CARD_W, 24 + CARD_H],
            radius=CARD_RADIUS,
            fill=CARD_FILL,
            outline=CARD_BORDER,
            width=1,
        )
        draw_engraved_rect(
            draw,
            cx + RECT_INSET,
            24 + RECT_INSET,
            CARD_W - 2 * RECT_INSET - MAX_OFFSET,
            CARD_H - 2 * RECT_INSET,
            UNCORRECTED_OFFSETS[col],
            corrected,
        )
        draw.text(
            (cx + CARD_W / 2, 24 + CARD_H + 12),
            f"{SPEEDS[col]} mm/min",
            fill=MUTED,
            font=label_font,
            anchor="ma",
        )
    return img


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for corrected, name in (
        (False, "bidir-scan-offset-uncorrected.webp"),
        (True, "bidir-scan-offset-table-applied.webp"),
    ):
        path = OUTPUT_DIR / name
        render(corrected).save(path, format="WEBP", lossless=True)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
