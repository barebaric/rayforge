"""Generate the bidirectional scan offset diagrams for the website docs.

Renders how right-to-left raster passes misalign by an amount that
grows with engrave speed, and how the machine's speed-dependent offset
table straightens the edges back out, into two images under
``website/static/images/``. The images carry no translatable words;
titles live in the markdown that embeds them.
"""

from PIL import Image, ImageDraw

from . import canvas, style

SPEEDS = [1000, 4000, 8000]
UNCORRECTED_OFFSETS = [6, 14, 24]
MAX_OFFSET = max(UNCORRECTED_OFFSETS)

CARD_W, CARD_H = 280, 170
CARD_GAP = 30
RECT_INSET = 16
LINE_PITCH = 8
LINE_THICKNESS = 3


def draw_engraved_rect(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    offset: int,
    corrected: bool,
) -> None:
    canvas.draw_dashed_line(
        draw, (x - 3, y - 6), (x - 3, y + h + 6), fill=style.GUIDE
    )
    canvas.draw_dashed_line(
        draw, (x + w + 2, y - 6), (x + w + 2, y + h + 6), fill=style.GUIDE
    )
    for index, line_y in enumerate(range(y, y + h, LINE_PITCH)):
        shift = 0 if corrected or index % 2 == 0 else offset
        draw.rectangle(
            [x + shift, line_y, x + shift + w, line_y + LINE_THICKNESS],
            fill=style.LINE,
        )


def render(corrected: bool) -> Image.Image:
    label_font = style.find_font("DejaVuSans.ttf", 18)

    columns = len(SPEEDS)
    width = 24 + columns * CARD_W + (columns - 1) * CARD_GAP + 24
    height = 24 + CARD_H + 12 + 30
    img = Image.new("RGB", (width, height), style.BG)
    draw = ImageDraw.Draw(img)

    for col in range(columns):
        cx = 24 + col * (CARD_W + CARD_GAP)
        canvas.draw_card(draw, (cx, 24, cx + CARD_W, 24 + CARD_H))
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
            fill=style.MUTED,
            font=label_font,
            anchor="ma",
        )
    return img


def generate() -> list[tuple[str, Image.Image]]:
    return [
        ("bidir-scan-offset-uncorrected.webp", render(False)),
        ("bidir-scan-offset-table-applied.webp", render(True)),
    ]
