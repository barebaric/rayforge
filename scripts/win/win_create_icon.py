#!/usr/bin/env python3
from pathlib import Path
from PIL import Image

here = Path(__file__).parent.parent.parent
source_path = here / "website/static/images/favicon.png"
ico_path = here / "rayforge.ico"

sizes = [(256, 256), (64, 64), (48, 48), (32, 32), (16, 16)]
img = Image.open(source_path)

if img.mode != "RGBA":
    img = img.convert("RGBA")

icons = [img.resize(size, Image.Resampling.LANCZOS) for size in sizes]
icons[0].save(ico_path, format="ICO", sizes=sizes, append_images=icons[1:])
print("Icon generation complete.")
