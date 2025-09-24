import warnings
from typing import Tuple, Optional
from xml.etree import ElementTree as ET

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ..util import parse_length, to_mm

# A standard fallback conversion factor for pixel units. Corresponds to 96 DPI.
PPI: float = 96.0
"""Standard Pixels Per Inch, used for fallback conversions."""

MM_PER_PX: float = 25.4 / PPI
"""Conversion factor for pixels to millimeters, based on 96 PPI."""


def _get_margins_from_data(
    data: bytes,
) -> Tuple[float, float, float, float]:
    """
    Calculates content margins as ratios from raw SVG data using pyvips.
    Returns (left, top, right, bottom) margins as fractions of total size.
    """
    if not data:
        return 0.0, 0.0, 0.0, 0.0

    try:
        root = ET.fromstring(data)

        # 1. Get original dimensions to determine aspect ratio.
        w_str = root.get("width")
        h_str = root.get("height")
        if not w_str or not h_str:
            return 0.0, 0.0, 0.0, 0.0  # Cannot determine aspect ratio.

        orig_w, _ = parse_length(w_str)
        orig_h, _ = parse_length(h_str)

        if orig_w <= 0 or orig_h <= 0:
            return 0.0, 0.0, 0.0, 0.0

        aspect_ratio = orig_w / orig_h

        # 2. Calculate proportional dimensions for rendering.
        measurement_size = 4096.0  # Use a larger size for better precision
        if aspect_ratio > 1:  # Wider than tall
            render_w = measurement_size
            render_h = measurement_size / aspect_ratio
        else:  # Taller than wide or square
            render_h = measurement_size
            render_w = measurement_size * aspect_ratio

        # 3. Modify SVG for a large, proportional render.
        # DO NOT set preserveAspectRatio="none", as this causes distortion.
        root.set("width", f"{render_w}px")
        root.set("height", f"{render_h}px")

        # Create viewBox if it's missing, which is crucial for the renderer
        # to have a coordinate system.
        if not root.get("viewBox"):
            root.set("viewBox", f"0 0 {orig_w} {orig_h}")

        img = pyvips.Image.svgload_buffer(ET.tostring(root))
        if img.bands < 4:
            img = img.bandjoin(255)  # Ensure alpha channel for trimming

        # Create a sharp, binary mask from the alpha channel.
        alpha = img[3]
        mask = alpha > 0
        # Explicitly tell find_trim that the background color is 0.
        left, top, w, h = mask.find_trim(background=0)

        if w == 0 or h == 0:
            # No content found, so no margins
            return 0.0, 0.0, 0.0, 0.0

        # 4. Calculate margins as a ratio of the PROPORTIONAL render size.
        return (
            left / render_w,
            top / render_h,
            (render_w - (left + w)) / render_w,
            (render_h - (top + h)) / render_h,
        )
    except (pyvips.Error, ET.ParseError, ValueError):
        # Return zero margins if SVG is invalid or processing fails
        return 0.0, 0.0, 0.0, 0.0


def trim_svg(data: bytes) -> bytes:
    """
    Crops an SVG to its content by adjusting the viewBox attribute.

    This function renders the SVG to a bitmap, finds the bounding box of the
    non-transparent content, and calculates new viewBox and dimensions to
    effectively trim the empty space. The original aspect ratio of the
    content is preserved.

    Args:
        data: The raw SVG data in bytes.

    Returns:
        The raw bytes of the modified, trimmed SVG. Returns original data if
        no trimming is necessary or if the SVG is invalid.
    """
    margins = _get_margins_from_data(data)
    # If there's nothing to trim (within a small tolerance), return the
    # original data.
    if all(m < 1e-5 for m in margins):
        return data

    left, top, right, bottom = margins

    try:
        root = ET.fromstring(data)

        w_str = root.get("width")
        h_str = root.get("height")
        if not w_str or not h_str:
            return data  # Cannot proceed without dimensions

        w_val, w_unit = parse_length(w_str)
        h_val, h_unit = parse_length(h_str)

        vb_str = root.get("viewBox")
        if vb_str:
            vb_x, vb_y, vb_w, vb_h = map(float, vb_str.split())
        else:
            # If no viewBox, it's implicitly '0 0 width height'
            vb_x, vb_y, vb_w, vb_h = 0, 0, w_val, h_val

        # Calculate new viewBox based on margins
        new_vb_x = vb_x + (left * vb_w)
        new_vb_y = vb_y + (top * vb_h)
        new_vb_w = vb_w * (1 - left - right)
        new_vb_h = vb_h * (1 - top - bottom)

        if new_vb_w <= 0 or new_vb_h <= 0:
            return data  # Avoid creating an invalid SVG

        root.set("viewBox", f"{new_vb_x} {new_vb_y} {new_vb_w} {new_vb_h}")

        # Update width and height to reflect the trimmed size
        new_w_val = w_val * (1 - left - right)
        new_h_val = h_val * (1 - top - bottom)
        root.set("width", f"{new_w_val}{w_unit or 'px'}")
        root.set("height", f"{new_h_val}{h_unit or 'px'}")

        # The content should fill the new view, so set aspect ratio to none
        root.set("preserveAspectRatio", "none")

        return ET.tostring(root)

    except (ET.ParseError, ValueError):
        return data


def get_natural_size(data: bytes) -> Optional[Tuple[float, float]]:
    """
    Analyzes raw SVG data to extract its natural, untrimmed dimensions in mm.

    Args:
        data: The raw SVG data in bytes.

    Returns:
        A tuple of (width_mm, height_mm), or None if dimensions cannot be
        determined.
    """
    if not data:
        return None

    try:
        root = ET.fromstring(data)

        w_str = root.get("width")
        h_str = root.get("height")
        if not w_str or not h_str:
            return None

        w_val, w_unit = parse_length(w_str)
        h_val, h_unit = parse_length(h_str)

        # Use the MM_PER_PX constant for conversion
        width_mm = to_mm(w_val, w_unit, px_factor=MM_PER_PX)
        height_mm = to_mm(h_val, h_unit, px_factor=MM_PER_PX)

        return width_mm, height_mm

    except (ValueError, ET.ParseError):
        return None
