#!/usr/bin/env python3
"""Generate the Invert Image before/after example for the website docs.

Renders a grayscale, photo-like landscape and its inverted counterpart
-- the same image as produced by the Image Editing addon's Invert edit,
which flips every colour band and leaves alpha alone -- into two images
under ``website/static/images/``. The images carry no translatable
words; titles live in the markdown that embeds them.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "website" / "static" / "images"

WIDTH, HEIGHT = 640, 400
HORIZON = 235
WATERLINE = 330
SUN_X, SUN_Y = 448, 118
SUN_RADIUS = 26
GLOW_RADIUS = 110
SEED = 20261013


def sky(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the sky brightness with the sun and its glow."""
    top, bottom = 225.0, 135.0
    img = top + (bottom - top) * (y / HORIZON)
    distance = np.sqrt((x - SUN_X) ** 2 + (y - SUN_Y) ** 2)
    glow = np.clip(1.0 - distance / GLOW_RADIUS, 0.0, 1.0) ** 2
    img += 55.0 * glow
    disc = distance <= SUN_RADIUS
    img[disc] = 250.0
    return img


def ridge(
    x: np.ndarray, base: float, amplitude: float, roughness: float
) -> np.ndarray:
    """Return the ridge line height for each x coordinate."""
    shape = (
        np.sin(x * 0.011 + base) * 0.55
        + np.sin(x * 0.027 + base * 2.1) * 0.30
        + np.sin(x * 0.053 + base * 3.7) * 0.15
    )
    return base + shape * amplitude * roughness


def mountains(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Draw the far and near mountain ranges onto the image."""
    far = ridge(x, 205.0, 38.0, 1.0)
    near = ridge(x, 262.0, 46.0, 1.15)
    far_mask = (y > far) & (y <= HORIZON)
    near_mask = y > near
    img[far_mask] = 108.0
    img[near_mask] = 62.0
    img[far_mask & near_mask] = 62.0


def water(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the lake brightness with a reflection of the sun."""
    depth = (y - WATERLINE) / (HEIGHT - WATERLINE)
    img = 96.0 - 26.0 * depth
    distance = np.abs(x - SUN_X)
    shimmer = (np.sin(y * 1.7) * 0.5 + 0.5) * 6.0 + np.sin(
        x * 0.9 + y * 0.6
    ) * 3.0
    beam = np.exp(-((distance / 46.0) ** 2)) * (1.0 - depth) ** 0.7
    return img + beam * (26.0 + shimmer)


def grain(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    """Add mild photographic grain."""
    return img + rng.normal(0.0, 2.2, img.shape)


def render() -> Image.Image:
    """Render the example photo as an RGB image."""
    rng = np.random.default_rng(SEED)
    x, y = np.meshgrid(
        np.arange(WIDTH, dtype=float), np.arange(HEIGHT, dtype=float)
    )
    img = np.full((HEIGHT, WIDTH), 135.0)
    sky_mask = y < HORIZON
    img[sky_mask] = sky(x, y)[sky_mask]
    mountains(img, x, y)
    water_mask = y >= WATERLINE
    img[water_mask] = water(img, x, y)[water_mask]
    img = grain(rng, img)
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")
    pil = pil.filter(ImageFilter.GaussianBlur(0.4))
    return pil.convert("RGB")


def invert(image: Image.Image) -> Image.Image:
    """Invert the colour bands the way the Invert edit does."""
    return image.point(lambda value: 255 - value)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    before = render()
    for image, name in (
        (before, "image-edit-invert-before.webp"),
        (invert(before), "image-edit-invert-after.webp"),
    ):
        path = OUTPUT_DIR / name
        image.save(path, format="WEBP", lossless=True)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
