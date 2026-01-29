#!/usr/bin/env python3
"""Generate a video showing feature lines from features.md."""

import math
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


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


def create_feature_image(
    text: str,
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
    font_size: int = 120,
):
    """Create an image with the feature text centered."""
    img = Image.new("RGB", (width, height), color="#000000")
    draw = ImageDraw.Draw(img)

    font_path = find_font("DejaVuSans-Bold.ttf")
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    x = width // 2
    y = height // 2

    draw.text((x, y), text, fill="#ffffff", font=font, anchor="mm")
    img.save(output_path)


def generate_video(
    features_file: Path,
    output_video: Path,
    width: int = 1920,
    height: int = 1080,
    total_duration: float = 13.0,
    font_size: int = 130,
    first_item_duration: float = 1.5,
):
    """Generate a video from feature lines with accelerating pace."""
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)

    with open(features_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    n = len(lines)
    if n == 0:
        return

    durations = [first_item_duration]
    remaining = total_duration - first_item_duration
    remaining_items = n - 1

    if remaining_items > 0:
        weights = [math.exp(-0.1 * i) for i in range(remaining_items)]
        total_weight = sum(weights)
        for i in range(remaining_items - 1):
            t = remaining * weights[i] / total_weight
            durations.append(t)
            remaining -= t
            total_weight -= weights[i]
        durations.append(remaining)

    concat_file = temp_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for i, (line, duration) in enumerate(zip(lines, durations)):
            frame_path = temp_dir / f"frame_{i:04d}.png"
            create_feature_image(line, frame_path, width, height, font_size)
            abs_path = frame_path.resolve()
            f.write(f"file '{abs_path}'\n")
            f.write(f"duration {duration:.6f}\n")
        dummy_frame_path = temp_dir / "frame_dummy.png"
        create_feature_image("", dummy_frame_path, width, height, font_size)
        abs_dummy_path = dummy_frame_path.resolve()
        f.write(f"file '{abs_dummy_path}'\n")
        f.write("duration 0.001\n")

    output_video.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        f"scale={width}:{height}",
        "-r",
        "30",
        str(output_video),
    ]

    subprocess.run(cmd, check=True)

    for frame in temp_dir.glob("*.png"):
        frame.unlink()
    concat_file.unlink()
    if temp_dir.exists():
        temp_dir.rmdir()


if __name__ == "__main__":
    generate_video(
        Path("features.md"),
        Path("media/1.0/features.mp4"),
        width=1920,
        height=1080,
    )
