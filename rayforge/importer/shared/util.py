import math
import re

# Cairo has a hard limit on surface dimensions, often 32767.
# We use a slightly more conservative value to be safe.
CAIRO_MAX_DIMENSION = 16384


def to_mm(value, unit, px_factor=None):
    """Convert a value to millimeters based on its unit."""
    if unit == "cm":
        return value * 10
    if unit == "mm":
        return value
    if unit == "in":
        return value * 25.4
    if unit == "pt":
        return value * 25.4 / 72
    if px_factor and unit in ("", "px"):
        return value * px_factor
    raise ValueError("Cannot convert to millimeters without DPI information.")


def parse_length(s):
    if not s:
        return 0.0, "px"
    m = re.match(r"([0-9.]+)\s*([a-z%]*)", s)
    if m:
        return float(m.group(1)), m.group(2) or "px"
    return float(s), "px"


def calculate_chunk_layout(
    real_width, real_height, max_chunk_width, max_chunk_height, max_memory_size
):
    bytes_per_pixel = 4  # cairo.FORMAT_ARGB32

    effective_max_width = min(
        max_chunk_width
        if max_chunk_width is not None
        else CAIRO_MAX_DIMENSION,
        CAIRO_MAX_DIMENSION,
    )
    chunk_width = min(real_width, effective_max_width)

    possible_heights = [
        min(
            max_chunk_height
            if max_chunk_height is not None
            else CAIRO_MAX_DIMENSION,
            CAIRO_MAX_DIMENSION,
        )
    ]
    if max_memory_size is not None and chunk_width > 0:
        possible_heights.append(
            math.floor(max_memory_size / (chunk_width * bytes_per_pixel))
        )

    chunk_height = min(real_height, *possible_heights)
    chunk_width, chunk_height = max(1, chunk_width), max(1, chunk_height)

    return (
        chunk_width,
        math.ceil(real_width / chunk_width),
        chunk_height,
        math.ceil(real_height / chunk_height),
    )
