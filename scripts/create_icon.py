#!/usr/bin/env python3
"""Generate all icons and logos from the app icon SVG.

The single source of truth is
rayforge/resources/icons/org.rayforge.rayforge.svg. Since it may
contain transparent margins around the artwork, the content bounding
box is detected first and every derived asset is rendered against that
region:

- rayforge.icns and rayforge/resources/icons/rayforge.icns
  (macOS app icon; the icon is layered onto the squircle tile in
  rayforge/resources/icons/icon-app-tile.svg: a black, 40% opacity,
  blurred copy as shadow below the unchanged icon. Assembled directly,
  no rsvg-convert/iconutil needed)
- rayforge/resources/icons/rayforge.icon/Assets/icon.png
  (Icon Composer source image)
- website/static/images/icon.svg (verbatim copy for the website)
- website/static/images/icon-avatar.svg (margin-trimmed, for circular
  avatars such as the blog author thumbnail)
- website/static/images/icon.webp (website logo)
- website/static/images/favicon.png/.webp
- website/static/images/social.webp (icon centered on 1280x640)

Usage:
    python3 scripts/create_icon.py [output.icns]

Without an argument, both ICNS locations are updated.
"""

import io
import re
import struct
import sys
from pathlib import Path

import cairosvg
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_SVG_PATH = (
    PROJECT_ROOT / "rayforge/resources/icons/org.rayforge.rayforge.svg"
)
TILE_SVG_PATH = PROJECT_ROOT / "rayforge/resources/icons/icon-app-tile.svg"
ASSET_PNG_PATH = (
    PROJECT_ROOT / "rayforge/resources/icons/rayforge.icon/Assets/icon.png"
)
ICNS_PATHS = [
    PROJECT_ROOT / "rayforge.icns",
    PROJECT_ROOT / "rayforge/resources/icons/rayforge.icns",
]
WEB_SVG_PATH = PROJECT_ROOT / "website/static/images/icon.svg"
AVATAR_SVG_PATH = PROJECT_ROOT / "website/static/images/icon-avatar.svg"
WEBP_PATH = PROJECT_ROOT / "website/static/images/icon.webp"
FAVICON_PNG_PATH = PROJECT_ROOT / "website/static/images/favicon.png"
FAVICON_WEBP_PATH = PROJECT_ROOT / "website/static/images/favicon.webp"
SOCIAL_WEBP_PATH = PROJECT_ROOT / "website/static/images/social.webp"

TRIM_RENDER_SIZE = 1024
TILE_SVG_SIZE = 1024
AVATAR_PADDING = 0.1
ASSET_PNG_SIZE = 1024
WEBP_SIZE = 600
FAVICON_SIZE = 256
WEBP_QUALITY = 90
SOCIAL_SIZE = (1280, 640)
SOCIAL_QUALITY = 85

ICNS_SIZES = [
    ("icp4", 16),
    ("icp5", 32),
    ("ic07", 128),
    ("ic08", 256),
    ("ic09", 512),
    ("ic10", 1024),
]

TILE_ICON_FRACTION = 0.62
SHADOW_BLUR = 20
SHADOW_OPACITY = 0.4


def detect_trim(svg_text: str) -> tuple[float, float, float, float]:
    """Return the artwork region as (x, y, width, height) user units."""
    viewbox_match = re.search(r'viewBox="([^"]+)"', svg_text)
    if not viewbox_match:
        raise ValueError("Source SVG has no viewBox attribute")
    x, y, w, h = (float(v) for v in viewbox_match.group(1).split())

    png = cairosvg.svg2png(
        bytestring=svg_text.encode(),
        output_width=TRIM_RENDER_SIZE,
        output_height=TRIM_RENDER_SIZE,
    )
    bbox = Image.open(io.BytesIO(png)).getbbox()
    if bbox is None:
        return x, y, w, h

    scale = w / TRIM_RENDER_SIZE
    trim_x = x + bbox[0] * scale
    trim_y = y + bbox[1] * (h / TRIM_RENDER_SIZE)
    trim_w = (bbox[2] - bbox[0]) * scale
    trim_h = (bbox[3] - bbox[1]) * (h / TRIM_RENDER_SIZE)
    return trim_x, trim_y, trim_w, trim_h


def square_fit(
    trim: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Center the artwork in a square viewbox with AVATAR_PADDING
    margin on each side, so it sits nicely inside a circular avatar."""
    tx, ty, tw, th = trim
    side = max(tw, th) / (1 - 2 * AVATAR_PADDING)
    cx = tx + tw / 2
    cy = ty + th / 2
    return (cx - side / 2, cy - side / 2, side, side)


def apply_viewbox(
    svg_text: str, viewbox: tuple[float, float, float, float]
) -> str:
    values = " ".join(str(round(v, 2)) for v in viewbox)
    return re.sub(r'viewBox="[^"]+"', f'viewBox="{values}"', svg_text, count=1)


def build_icns(render_trimmed: callable) -> bytes:
    chunks = b""
    for os_type, size in ICNS_SIZES:
        png = build_tile_composite(render_trimmed, size)
        chunks += os_type.encode() + struct.pack(">I", len(png) + 8) + png
    return b"icns" + struct.pack(">I", len(chunks) + 8) + chunks


def build_tile_composite(render: callable, size: int) -> bytes:
    """Render the icon layered onto the squircle tile.

    Layer 1 is a black silhouette of the icon at reduced opacity with
    a gaussian blur; layer 2 is the unchanged icon on top.
    """
    tile_png = cairosvg.svg2png(
        url=str(TILE_SVG_PATH), output_width=size, output_height=size
    )
    canvas = Image.open(io.BytesIO(tile_png)).convert("RGBA")

    icon_size = round(size * TILE_ICON_FRACTION)
    offset = (size - icon_size) // 2

    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    silhouette = Image.new("RGBA", (icon_size, icon_size), (0, 0, 0, 255))
    with Image.open(io.BytesIO(render(icon_size))) as icon:
        alpha = icon.getchannel("A")
        shadow.paste(silhouette, (offset, offset), alpha)

    blur_radius = SHADOW_BLUR * size / TILE_SVG_SIZE
    alpha = shadow.getchannel("A").point(lambda v: int(v * SHADOW_OPACITY))
    shadow.putalpha(alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
    canvas.alpha_composite(shadow)

    with Image.open(io.BytesIO(render(icon_size))) as icon:
        canvas.alpha_composite(icon.convert("RGBA"), (offset, offset))

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def png_to_webp(png_data: bytes, quality: int = WEBP_QUALITY) -> bytes:
    with Image.open(io.BytesIO(png_data)) as image:
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=quality)
        return buffer.getvalue()


def build_social_webp(render: callable) -> bytes:
    width, height = SOCIAL_SIZE
    icon_png = render(height)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    with Image.open(io.BytesIO(icon_png)) as icon:
        canvas.paste(icon, ((width - height) // 2, 0), icon)
    buffer = io.BytesIO()
    canvas.save(buffer, format="WEBP", quality=SOCIAL_QUALITY)
    return buffer.getvalue()


def write(path: Path, data: bytes) -> None:
    print(f"Writing {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> int:
    if not SOURCE_SVG_PATH.exists():
        print(
            f"Error: SVG file not found at {SOURCE_SVG_PATH}", file=sys.stderr
        )
        return 1

    svg_text = SOURCE_SVG_PATH.read_text()
    trim = detect_trim(svg_text)
    trimmed_svg = apply_viewbox(svg_text, trim)

    def render(size: int) -> bytes:
        return cairosvg.svg2png(
            bytestring=svg_text.encode(), output_width=size, output_height=size
        )

    def render_trimmed(size: int) -> bytes:
        return cairosvg.svg2png(
            bytestring=trimmed_svg.encode(),
            output_width=size,
            output_height=size,
        )

    icns_data = build_icns(render)
    favicon_png = render_trimmed(FAVICON_SIZE)

    for path in ICNS_PATHS:
        write(path, icns_data)
    write(ASSET_PNG_PATH, render(ASSET_PNG_SIZE))
    write(WEB_SVG_PATH, svg_text.encode())
    avatar_viewbox = square_fit(trim)
    write(AVATAR_SVG_PATH, apply_viewbox(svg_text, avatar_viewbox).encode())
    write(WEBP_PATH, png_to_webp(render(WEBP_SIZE)))
    write(FAVICON_PNG_PATH, favicon_png)
    write(FAVICON_WEBP_PATH, png_to_webp(favicon_png))
    write(SOCIAL_WEBP_PATH, build_social_webp(render))

    size_mb = len(icns_data) / (1024 * 1024)
    print(f"\nDone! ICNS created ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
