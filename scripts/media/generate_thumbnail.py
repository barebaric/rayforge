#!/usr/bin/env python3
"""Generate release thumbnails using AI backgrounds and version text."""

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow not installed. Install with: pip install Pillow")
    raise


def find_font(font_name: str = "DejaVuSans-Bold.ttf") -> str | None:
    """Find a font file from common system locations."""
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


def parse_version(version_str):
    """Parse version string into components."""
    match = re.match(r"(\d+)(?:\.(\d+))?", version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")
    major = match.group(1)
    minor = match.group(2) or ""
    return major, minor


def get_app_logo():
    """Get path to app logo."""
    logo_path = (
        Path(__file__).parent.parent
        / "rayforge"
        / "resources"
        / "icons"
        / "org.rayforge.rayforge.svg"
    )
    if not logo_path.exists():
        logo_path = Path(__file__).parent.parent / "media" / "fiber-laser.png"
    return logo_path


def get_previous_thumbnail(release_dir):
    """Get path to previous release thumbnail."""
    media_dir = Path(__file__).parent.parent / "media"
    if not media_dir.exists():
        return None

    release_dirs = sorted(
        [d for d in media_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name,
        reverse=True,
    )

    for release in release_dirs:
        if release == release_dir:
            continue
        thumbnail = release / "thumbnail.png"
        if thumbnail.exists():
            return thumbnail
    return None


def generate_ai_background(version, seed=None):
    """Generate AI background using comfyui MCP server."""
    if seed is None:
        seed = random.randint(1, 1000000)

    prompts = [
        "laser cutting workspace, professional CNC machine, "
        "modern dark theme, version {v} software interface, "
        "high contrast, cinematic lighting, 1920x1080, "
        "8k quality, sharp focus",
        "abstract laser beam patterns, geometric shapes, "
        "glowing blue and orange lines, dark background, "
        "version {v} overlay style, futuristic tech, "
        "cinematic, high contrast",
        "precision engineering workspace, CAD software "
        "interface, blueprints, technical drawings, dark "
        "mode, version {v} branding, professional, "
        "clean design, 8k",
        "laser engraving close-up, sparks flying, metal "
        "texture, dark industrial setting, version {v} "
        "watermark, dramatic lighting, high detail, "
        "1920x1080",
        "modern software interface, dark theme, blue accent "
        "colors, geometric patterns, version {v} hero image, "
        "minimalist design, clean background, "
        "professional software",
    ]

    prompt = random.choice(prompts).format(v=version)

    mcp_script = f"""import json
import sys

result = {{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {{
        "name": "mcp--comfyui--generate_image",
        "arguments": {{
            "prompt": {json.dumps(prompt)},
            "width": 1920,
            "height": 1080,
            "steps": 8,
            "cfg": 2,
            "seed": {seed}
        }}
    }}
}}

print(json.dumps(result))
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", mcp_script],
            capture_output=True,
            text=True,
            check=True,
        )
        output = json.loads(result.stdout)
        if "result" in output:
            image_path = output["result"].get("content", [{}])[0].get("text")
            if image_path:
                return Path(image_path)
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        pass

    return None


def create_thumbnail(version, output_path, base_image=None, use_ai=True):
    """Create a release thumbnail with version number."""
    width, height = 1920, 1080

    if use_ai and not base_image:
        ai_image = generate_ai_background(version)
        if ai_image and ai_image.exists():
            base_image = ai_image

    if base_image and base_image.exists():
        img = Image.open(base_image).convert("RGBA")
        img = img.resize((width, height), Image.Resampling.LANCZOS)
    else:
        img = Image.new("RGBA", (width, height), (51, 51, 51, 255))

    draw = ImageDraw.Draw(img)

    major, minor = parse_version(version)

    font_path = find_font("DejaVuSans-Bold.ttf")
    if font_path:
        try:
            font_large = ImageFont.truetype(font_path, 180)
            font_small = ImageFont.truetype(font_path, 80)
        except OSError:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
    else:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    version_text = f"v{major}"
    if minor:
        version_text += f".{minor}"

    bbox = draw.textbbox((0, 0), version_text, font=font_large)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), version_text, fill=(255, 255, 255, 255), font=font_large)

    sub_text = "Rayforge"
    bbox_sub = draw.textbbox((0, 0), sub_text, font=font_small)
    sub_width = bbox_sub[2] - bbox_sub[0]
    sub_x = (width - sub_width) // 2
    sub_y = y - 100

    draw.text(
        (sub_x, sub_y),
        sub_text,
        fill=(200, 200, 200, 255),
        font=font_small,
    )

    if minor:
        minor_text = f".{minor}"
        minor_x = x + text_width + 10
        minor_y = y + text_height - 80

        draw.text(
            (minor_x, minor_y),
            minor_text,
            fill=(100, 200, 255, 255),
            font=font_small,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    print(f"Thumbnail saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate release thumbnail with version number"
    )
    parser.add_argument(
        "version",
        help="Release version (e.g., '1.0' or '0.24')",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=1,
        help="Starting thumbnail number (default: 1)",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=1,
        help="Number of thumbnails to generate (default: 1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for thumbnail "
        "(default: media/<version>/drafts/thumbnail<number>.png)",
    )
    parser.add_argument(
        "-b",
        "--base",
        help="Base image to use as background "
        "(default: AI-generated or app logo)",
    )
    parser.add_argument(
        "--use-previous",
        action="store_true",
        help="Use previous release thumbnail as base",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI background generation",
    )

    args = parser.parse_args()

    base_image = None
    if args.base:
        base_image = Path(args.base)
    elif args.use_previous:
        base_dir = Path(__file__).parent.parent / "media" / args.version
        base_image = get_previous_thumbnail(base_dir)
        if base_image:
            print(f"Using previous thumbnail: {base_image}")

    use_ai = not args.no_ai and not base_image

    for i in range(args.count):
        thumb_num = args.number + i

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = (
                Path(__file__).parent.parent
                / "media"
                / args.version
                / "drafts"
                / f"thumbnail{thumb_num}.png"
            )

        create_thumbnail(args.version, output_path, base_image, use_ai)


if __name__ == "__main__":
    main()
