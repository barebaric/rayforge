#!/usr/bin/env python3
"""
Minimize raster images by converting them to WebP.

Converts raster images (PNG, JPEG, TIFF, BMP) in a directory to WebP,
which is typically several times smaller while staying visually
lossless at reasonable quality levels.

Original files are never modified or deleted by default; use
--remove-original to delete them after a successful conversion.

WebP outputs are regenerated from their sources whenever a source
(e.g. hero.png) is present - outputs are treated as regenerable
artifacts. WebP files that have no source are left untouched, since
re-encoding them would be a lossy re-encode of an original; use
--reprocess-webp to opt into that. Never touches SVGs.

Example:
    pixi run python scripts/optimize_images.py website/static/images
    pixi run python scripts/optimize_images.py website/static/images \
        --quality 85 --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

import pyvips

CONVERTIBLE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

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


def convert_image(
    src: Path,
    dst: Path,
    quality: int,
    remove_original: bool,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Convert one image file to WebP.

    Returns (old_size, new_size).
    """
    old_size = src.stat().st_size
    action = "Re-processing" if src == dst else "Converting"
    logger.info("%s %s (%s)", action, src.name, format_size(old_size))

    if dry_run:
        return old_size, 0

    image = pyvips.Image.new_from_file(str(src))
    image = image.colourspace("srgb")

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
    if new_size >= old_size:
        logger.info(
            "  %s: WebP not smaller (%s -> %s), keeping original",
            src.name,
            format_size(old_size),
            format_size(new_size),
        )
        out_path.unlink()
        return old_size, old_size

    out_path.replace(dst)
    if src != dst and remove_original:
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
    remove_original: bool,
    reprocess_webp: bool,
    dry_run: bool,
) -> int:
    sources = find_source_files(directory)
    webps = find_webp_files(directory)
    if not sources and not webps:
        logger.info("No convertible images found in %s", directory)
        return 0

    total_old = total_new = 0
    failures = 0
    jobs = [(src, src.with_suffix(".webp")) for src in sources]

    # WebP files without a source are originals; re-encoding them is
    # lossy and only done on explicit request.
    source_stems = {p.stem for p in sources}
    if reprocess_webp:
        jobs.extend((webp, webp) for webp in webps)
    elif webps:
        orphans = [webp for webp in webps if webp.stem not in source_stems]
        if orphans:
            logger.info(
                "Skipping %d WebP file(s) without a source file "
                "(use --reprocess-webp to re-encode them lossily)",
                len(orphans),
            )

    for src, dst in jobs:
        try:
            old_size, new_size = convert_image(
                src,
                dst,
                quality,
                remove_original,
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
        description="Minimize images by converting them to WebP."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing the images to optimize",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=90,
        help="WebP quality (default: 90)",
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
        args.remove_original,
        args.reprocess_webp,
        args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
