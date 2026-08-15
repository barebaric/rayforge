#!/usr/bin/env python3
"""
Minimize texture images in material libraries.

Converts raster texture images (PNG, JPEG, TIFF, BMP) in a materials
directory to WebP, which is roughly 10-30x smaller for photographic
textures while staying visually lossless and fast to decode.

By default each texture is also normalized to a square target size
(1000x1000): larger images are center-cropped, smaller images are
repeated to fill the target and then cropped. After normalization the
image is made seamlessly tileable by cross-fading it with half-offset
copies of itself (per axis, blended in linear light), so it can be
repeated via GL_REPEAT without visible seams.

Original files are never modified or deleted by default; use
--remove-original to delete them after a successful conversion.
Material YAML files that reference a converted file by name (e.g.
"appearance.texture: oak.png") are updated to the new filename,
preserving all other formatting.

WebP outputs are regenerated from their sources whenever a source
(e.g. oak.png) is present - outputs are treated as regenerable
artifacts. WebP files that have no source are left untouched, since
re-encoding them would be a lossy re-encode of an original; use
--reprocess-webp to opt into that. Never touches SVGs.

Example:
    pixi run python scripts/optimize_material_textures.py
    pixi run python scripts/optimize_material_textures.py \
        --quality 90 --size 512 --dry-run
"""

import argparse
import logging
import math
import re
import sys
from pathlib import Path

import numpy as np
import pyvips

DEFAULT_TARGET = (
    Path(__file__).parent.parent
    / "rayforge"
    / "builtin_addons"
    / "rayforge-addon-materials"
    / "materials"
)

CONVERTIBLE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
YAML_EXTENSIONS = {".yaml", ".yml"}

logger = logging.getLogger(__name__)


def format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MiB"
    return f"{num_bytes / 1024:.0f} KiB"


def find_source_files(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.suffix.lower() in CONVERTIBLE_EXTENSIONS and p.is_file()
    )


def find_webp_files(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.suffix.lower() == ".webp" and p.is_file()
    )


def find_yaml_files(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.suffix.lower() in YAML_EXTENSIONS and p.is_file()
    )


def update_yaml_references(
    yaml_files: list[Path], old_name: str, new_name: str
) -> list[Path]:
    """
    Replace occurrences of old_name with new_name in the given YAML
    files, preserving formatting. Returns the files that changed.
    """
    pattern = re.compile(re.escape(old_name))
    changed = []
    for yaml_path in yaml_files:
        text = yaml_path.read_text(encoding="utf-8")
        if not pattern.search(text):
            continue
        yaml_path.write_text(pattern.sub(new_name, text), encoding="utf-8")
        changed.append(yaml_path)
    return changed


def normalize_size(image: pyvips.Image, size: int) -> pyvips.Image:
    """
    Normalize the image to a size x size square.

    Larger images are center-cropped to a square around their
    smaller dimension and scaled down to size (equivalent to scaling
    so the smaller dimension equals size, then cropping; no content
    of the smaller dimension is lost). Smaller images are repeated to
    fill the target and then cropped (the caller makes the result
    seamlessly tileable afterwards, which also hides the repetition
    seams).
    """
    height, width = image.height, image.width
    if height == size and width == size:
        return image

    if width >= size and height >= size:
        side = min(width, height)
        x = (width - side) // 2
        y = (height - side) // 2
        if side != width or side != height:
            logger.info("  cropping %dx%d -> %dx%d", width, height, side, side)
            image = image.crop(x, y, side, side)
        if side != size:
            logger.info("  scaling %dx%d -> %dx%d", side, side, size, size)
            image = image.resize(size / side)
        return image

    arr = np.frombuffer(image.write_to_memory(), dtype=np.uint8)
    arr = arr.reshape(height, width, image.bands)
    reps = (
        math.ceil(size / height),
        math.ceil(size / width),
        1,
    )
    tiled = np.tile(arr, reps)[:size, :size]
    logger.info(
        "  tiled %dx%d up to %dx%d",
        width,
        height,
        tiled.shape[1],
        tiled.shape[0],
    )
    return pyvips.Image.new_from_array(np.ascontiguousarray(tiled))


def _cosine_ramp(n: int) -> np.ndarray:
    """
    1D blend weight along one axis: 0 at the borders, 1 at the
    center, with a raised-cosine profile so the cross-fade has no
    derivative discontinuities (avoids Mach banding).
    """
    x = np.arange(n, dtype=np.float32)
    s = np.clip(np.minimum(x + 0.5, n - x - 0.5) / (n / 2), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * s)


def make_tileable(image: pyvips.Image) -> pyvips.Image:
    """
    Make an image seamlessly tileable.

    For each axis independently, the image is cross-faded with a copy
    of itself shifted by half the axis length, weighted by a cosine
    ramp that is 0 at the borders and 1 at the center. Opposite
    edges of the output therefore come from adjacent pixels of the
    source interior, so the texture repeats without visible seams.

    The passes are separable (one axis at a time) on purpose: the
    shifted copy has its own wrap discontinuity along the axis
    midpoint, and blending with that axis's ramp suppresses it
    everywhere along the midline. A single diagonal shift with a
    corner mask would leak that discontinuity as a visible seam
    along the middle of the image. Blending happens in linear light
    to avoid brightness halos.
    """
    height, width = image.height, image.width
    if width < 4 or height < 4:
        return image

    # Blend colour bands only; alpha would blend nonsensically.
    if image.bands > 3:
        image = image.extract_band(0, n=3)

    linear = image.colourspace("scrgb")
    dtype = np.float32 if linear.format == "float" else np.uint8
    arr = np.frombuffer(linear.write_to_memory(), dtype=dtype)
    arr = arr.reshape(height, width, linear.bands).astype(np.float32)

    for axis, n in ((0, height), (1, width)):
        shifted = np.roll(arr, n // 2, axis=axis)
        ramp = _cosine_ramp(n)
        weight = ramp.reshape([-1, 1, 1] if axis == 0 else [1, -1, 1])
        arr = arr * weight + shifted * (1.0 - weight)

    out = pyvips.Image.new_from_array(np.ascontiguousarray(arr))
    return out.copy(interpretation="scrgb").colourspace("srgb")


def convert_texture(
    src: Path,
    dst: Path,
    quality: int,
    size: int,
    tileable: bool,
    remove_original: bool,
    yaml_files: list[Path],
    dry_run: bool,
) -> tuple[int, int]:
    """
    Convert one image file to WebP and update YAML references.

    Returns (old_size, new_size).
    """
    old_size = src.stat().st_size
    action = "Re-processing" if src == dst else "Converting"
    logger.info("%s %s (%s)", action, src.name, format_size(old_size))

    if dry_run:
        return old_size, 0

    image = pyvips.Image.new_from_file(str(src))
    image = image.colourspace("srgb")
    image = normalize_size(image, size)
    if tileable:
        image = make_tileable(image)
    if not image.hasalpha():
        image = image.addalpha()

    # Encode to a temporary path first so an in-place re-encode of an
    # existing .webp cannot destroy the input on failure.
    out_path = dst.with_name(dst.name + ".tmp")
    try:
        # effort=6 is the slowest/best compression; one-off tool, so
        # encode time is irrelevant.
        image.webpsave(str(out_path), Q=quality, effort=6)
    except pyvips.error.Error:
        logger.error("Failed to encode %s, skipping", src.name)
        if out_path.exists():
            out_path.unlink()
        raise

    # Round-trip check before touching the original.
    reloaded = pyvips.Image.new_from_file(str(out_path))
    if (reloaded.width, reloaded.height) != (image.width, image.height):
        logger.error(
            "Round-trip check failed for %s (size mismatch), keeping original",
            src.name,
        )
        out_path.unlink()
        raise ValueError(f"round-trip mismatch for {src.name}")

    new_size = out_path.stat().st_size
    if not tileable and new_size >= old_size:
        logger.info(
            "  %s: WebP not smaller (%s -> %s), keeping original",
            src.name,
            format_size(old_size),
            format_size(new_size),
        )
        out_path.unlink()
        return old_size, old_size

    out_path.replace(dst)
    if src != dst and src.name != dst.name:
        changed_yamls = update_yaml_references(yaml_files, src.name, dst.name)
        for yaml_path in changed_yamls:
            logger.info("  updated reference in %s", yaml_path.name)

    if remove_original and src != dst:
        src.unlink()
        logger.info(
            "  %s -> %s (%.1fx smaller, original removed)",
            format_size(old_size),
            format_size(new_size),
            old_size / max(new_size, 1),
        )
    else:
        note = ", original kept" if src != dst else ""
        logger.info(
            "  %s -> %s (%.1fx smaller%s)",
            format_size(old_size),
            format_size(new_size),
            old_size / max(new_size, 1),
            note,
        )
    return old_size, new_size


def optimize_directory(
    directory: Path,
    quality: int,
    size: int,
    tileable: bool,
    remove_original: bool,
    reprocess_webp: bool,
    dry_run: bool,
) -> int:
    sources = find_source_files(directory)
    yaml_files = find_yaml_files(directory)
    webps = find_webp_files(directory)
    if not sources and not webps:
        logger.info("No convertible textures found in %s", directory)
        return 0

    total_old = total_new = 0
    failures = 0
    for src in sources:
        try:
            old_size, new_size = convert_texture(
                src,
                src.with_suffix(".webp"),
                quality,
                size,
                tileable,
                remove_original,
                yaml_files,
                dry_run,
            )
        except (pyvips.error.Error, ValueError) as exc:
            logger.error("  %s", exc)
            failures += 1
            continue
        total_old += old_size
        total_new += new_size

    # WebP files without a source are originals; re-encoding them is
    # lossy and only done on explicit request.
    source_stems = {p.stem for p in sources}
    for webp in webps:
        if webp.stem in source_stems:
            continue
        if not reprocess_webp:
            logger.info(
                "Skipping %s: WebP without a source file "
                "(use --reprocess-webp to re-encode it lossily)",
                webp.name,
            )
            continue
        try:
            old_size, new_size = convert_texture(
                webp,
                webp,
                quality,
                size,
                tileable,
                remove_original,
                yaml_files,
                dry_run,
            )
        except (pyvips.error.Error, ValueError) as exc:
            logger.error("  %s", exc)
            failures += 1
            continue
        total_old += old_size
        total_new += new_size

    if total_new:
        logger.info(
            "Total: %s -> %s (%.1fx smaller)",
            format_size(total_old),
            format_size(total_new),
            total_old / max(total_new, 1),
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Minimize material textures by converting them to WebP."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Materials directory (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=80,
        help="WebP quality (default: 80)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1000,
        help="Target square size in pixels: larger images are "
        "center-cropped, smaller ones are tiled up to it "
        "(default: 1000)",
    )
    parser.add_argument(
        "--no-tileable",
        dest="tileable",
        action="store_false",
        help="Do not make textures seamlessly tileable (tiling is "
        "on by default)",
    )
    parser.add_argument(
        "--remove-original",
        action="store_true",
        help="Delete original files after a successful conversion "
        "(originals are kept by default)",
    )
    parser.add_argument(
        "--reprocess-webp",
        action="store_true",
        help="Also re-encode WebP files that have no source file. "
        "This is a lossy re-encode of an original; prefer keeping "
        "sources and regenerating from them instead",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be converted without changing anything",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    logging.getLogger("pyvips").setLevel(logging.WARNING)

    if not args.directory.is_dir():
        logger.error("Not a directory: %s", args.directory)
        return 1

    return optimize_directory(
        args.directory,
        args.quality,
        args.size,
        args.tileable,
        args.remove_original,
        args.reprocess_webp,
        args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
