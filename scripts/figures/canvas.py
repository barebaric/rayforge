"""Drawing and output helpers shared by the figure generators."""

from pathlib import Path

from PIL import Image, ImageDraw

from . import style

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "website" / "static" / "images"

CARD_RADIUS = 12


def draw_card(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]
) -> None:
    draw.rounded_rectangle(
        box,
        radius=CARD_RADIUS,
        fill=style.CARD_FILL,
        outline=style.CARD_BORDER,
        width=1,
    )


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: str,
    dash: float = 5,
    gap: float = 4,
    width: int = 1,
) -> None:
    total = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    if total == 0:
        return
    unit = ((end[0] - start[0]) / total, (end[1] - start[1]) / total)
    travelled = 0.0
    while travelled < total:
        seg_end = min(travelled + dash, total)
        point_a = (
            start[0] + unit[0] * travelled,
            start[1] + unit[1] * travelled,
        )
        point_b = (
            start[0] + unit[0] * seg_end,
            start[1] + unit[1] * seg_end,
        )
        draw.line([point_a, point_b], fill=fill, width=width)
        travelled = seg_end + gap


def draw_arrow_head(
    draw: ImageDraw.ImageDraw,
    tip: tuple[float, float],
    direction: tuple[float, float],
    fill: str,
    size: float = 7,
) -> None:
    length = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
    if length == 0:
        return
    unit = (direction[0] / length, direction[1] / length)
    back = (-unit[0], -unit[1])
    left = (-unit[1], unit[0])
    base = (tip[0] + back[0] * size, tip[1] + back[1] * size)
    wing_a = (
        base[0] + left[0] * size * 0.5,
        base[1] + left[1] * size * 0.5,
    )
    wing_b = (
        base[0] - left[0] * size * 0.5,
        base[1] - left[1] * size * 0.5,
    )
    draw.polygon([tip, wing_a, wing_b], fill=fill)


def save_webp(filename: str, image: Image.Image) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    image.save(path, format="WEBP", lossless=True)
    print(f"Saved {path}")
    return path
