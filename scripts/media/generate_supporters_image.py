#!/usr/bin/env python3
"""Generate a thank-you image for supporters to use in videos."""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
BG_COLOR = (18, 18, 24)
ACCENT_COLOR = (100, 180, 255)
TEXT_COLOR = (240, 240, 245)
SUBTLE_COLOR = (140, 140, 160)

SUPPORTERS_FILE = Path(__file__).resolve().parent.parent.parent / (
    "media/supporters.md"
)

SECTION_NAMED = "## Agreed to be mentioned"
SECTION_ANONYMOUS = (
    '## Did **not** agree to be mentioned (should be mentioned as "anonymous")'
)


def find_font(font_name: str = "DejaVuSans-Bold.ttf") -> Optional[str]:
    font_paths = [
        Path("/usr/share/fonts/truetype/dejavu") / font_name,
        Path("/usr/share/fonts/truetype/liberation") / font_name,
        Path("/usr/share/fonts/truetype/freefont") / font_name,
        Path.home() / ".local" / "share" / "fonts" / font_name,
        Path("/System/Library/Fonts") / font_name,
        Path("C:\\Windows\\Fonts") / font_name,
    ]
    for path in font_paths:
        if path.exists():
            return str(path)
    return None


def parse_supporters(filepath: Path) -> Tuple[List[str], int]:
    content = filepath.read_text()
    lines = content.splitlines()

    named_names = []
    anonymous_count = 0
    current_section = None

    for line in lines:
        stripped = line.strip()
        if stripped == SECTION_NAMED:
            current_section = "named"
            continue
        elif stripped == SECTION_ANONYMOUS:
            current_section = "anonymous"
            continue
        elif stripped.startswith("# "):
            current_section = None
            continue

        if not stripped:
            continue

        m = re.match(r"^\d{4}-\d{2}-\d{2}\s+(.+?)(?:\s+\(.+\))?$", stripped)
        if m:
            if current_section == "named":
                named_names.append(m.group(1).strip())
            elif current_section == "anonymous":
                anonymous_count += 1

    return named_names, anonymous_count


def generate_image(
    names: List[str],
    anonymous_count: int,
    output_path: Path,
    title: str = "Thank You",
    subtitle: str = "to our supporters",
):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)  # type: ignore[arg-type]
    draw = ImageDraw.Draw(img)

    bold_path = find_font("DejaVuSans-Bold.ttf")
    regular_path = find_font("DejaVuSans.ttf")

    try:
        font_title = (
            ImageFont.truetype(bold_path, 100)
            if bold_path
            else ImageFont.load_default()
        )
        font_subtitle = (
            ImageFont.truetype(regular_path, 40)
            if regular_path
            else ImageFont.load_default()
        )
        font_name = (
            ImageFont.truetype(regular_path, 44)
            if regular_path
            else ImageFont.load_default()
        )
        font_name_bold = (
            ImageFont.truetype(bold_path, 44)
            if bold_path
            else ImageFont.load_default()
        )
        font_anon = (
            ImageFont.truetype(regular_path, 32)
            if regular_path
            else ImageFont.load_default()
        )
    except OSError:
        font_title = ImageFont.load_default()
        font_subtitle = font_title
        font_name = font_title
        font_name_bold = font_title
        font_anon = font_title

    line_w = WIDTH - 300
    cx = WIDTH // 2
    y_cursor = 80

    bbox = draw.textbbox((0, 0), title, font=font_title)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        ((WIDTH - tw) // 2, y_cursor),
        title,
        fill=ACCENT_COLOR,
        font=font_title,
    )
    y_cursor += th + 50

    bbox = draw.textbbox((0, 0), subtitle, font=font_subtitle)
    sw = bbox[2] - bbox[0]
    draw.text(
        ((WIDTH - sw) // 2, y_cursor),
        subtitle,
        fill=SUBTLE_COLOR,
        font=font_subtitle,
    )
    y_cursor += 100

    cols = _calc_columns(len(names))
    col_width = line_w // cols
    col_start_x = cx - line_w // 2

    for idx, name in enumerate(names):
        col = idx % cols
        row = idx // cols
        x = col_start_x + col * col_width + col_width // 2
        y = y_cursor + row * 58

        if y + 58 > HEIGHT - 120:
            break

        font = font_name_bold if col == 0 else font_name
        draw.text((x, y), name, fill=TEXT_COLOR, font=font, anchor="mt")

    total_rows = -(-len(names) // cols)
    y_cursor += total_rows * 58 + 50

    if anonymous_count > 0:
        anon_text = "...and to everyone who supported anonymously"
        bbox = draw.textbbox((0, 0), anon_text, font=font_anon)
        aw = bbox[2] - bbox[0]
        draw.text(
            ((WIDTH - aw) // 2, y_cursor),
            anon_text,
            fill=SUBTLE_COLOR,
            font=font_anon,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    print(f"Saved supporters image to: {output_path}")


def _calc_columns(count: int) -> int:
    if count <= 5:
        return 1
    elif count <= 12:
        return 2
    elif count <= 24:
        return 3
    return 4


def main():
    parser = argparse.ArgumentParser(
        description="Generate a thank-you image for supporters"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default: media/supporters.png)",
    )
    parser.add_argument(
        "-t",
        "--title",
        default="Thank You",
        help="Title text (default: 'Thank You')",
    )
    parser.add_argument(
        "-s",
        "--subtitle",
        default="To everyone supporting Rayforge",
        help="Subtitle text (default: 'To everyone supporting Rayforge')",
    )
    parser.add_argument(
        "-f",
        "--file",
        default=None,
        help=f"Path to supporters.md (default: {SUPPORTERS_FILE})",
    )
    args = parser.parse_args()

    filepath = Path(args.file) if args.file else SUPPORTERS_FILE
    if not filepath.exists():
        print(f"Error: {filepath} not found.")
        return

    names, anonymous_count = parse_supporters(filepath)
    if not names:
        print("No named supporters found.")
        return

    output = (
        Path(args.output)
        if args.output
        else filepath.parent / "supporters.png"
    )

    generate_image(
        names,
        anonymous_count,
        output,
        title=args.title,
        subtitle=args.subtitle,
    )


if __name__ == "__main__":
    main()
