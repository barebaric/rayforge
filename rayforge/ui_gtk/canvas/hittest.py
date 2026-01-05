from __future__ import annotations
from typing import TYPE_CHECKING
import cairo

if TYPE_CHECKING:
    from ...core.matrix import Matrix


def check_pixel_hit(
    surface: cairo.ImageSurface,
    content_transform: Matrix,
    element_width: float,
    element_height: float,
    hit_distance: float,
    local_x: float,
    local_y: float,
) -> bool:
    """
    Checks if a pixel at specific local coordinates is opaque on a cairo
    surface.

    This function is a utility for performing pixel-perfect hit-testing on a
    rendered element buffer. It accounts for an element's content transform
    and a "fuzzy" hit distance specified in screen pixels.

    Args:
        surface: The cairo ImageSurface to check against.
        content_transform: The transformation from the element's geometry
            space to its content space.
        element_width: The geometric width of the element.
        element_height: The geometric height of the element.
        hit_distance: A "fuzzy" radius in screen pixels. If > 0, checks a
            circular area for any opaque pixel.
        local_x: The x-coordinate in the element's local GEOMETRY space.
        local_y: The y-coordinate in the element's local GEOMETRY space.

    Returns:
        True if the pixel (or surrounding area) is considered a hit.
    """
    surface_w = surface.get_width()
    surface_h = surface.get_height()
    if surface_w <= 0 or surface_h <= 0:
        return False

    # The received coordinates are in the element's GEOMETRIC space.
    # We must transform them into the CONTENT's space before sampling
    # the surface.
    content_x, content_y = local_x, local_y
    if not content_transform.is_identity():
        try:
            # We need to map the geometric point to the content's
            # coordinate system. This requires the inverse of the
            # content_transform.
            inv_content = content_transform.invert()
            content_x, content_y = inv_content.transform_point(
                (local_x, local_y)
            )
        except Exception:
            # If matrix is non-invertible, we can't do the hit-test.
            # Default to a hit, as the user is inside the bounding box.
            return True

    # Scale CONTENT coordinates to surface pixel coordinates.
    center_surface_x = int(content_x * (surface_w / element_width))
    center_surface_y = int(content_y * (surface_h / element_height))

    # Clamp the calculated pixel coordinates to be safely within the
    # surface bounds [0, dim-1]. This guards against floating-point
    # inaccuracies where a coordinate might be calculated to be exactly
    # the surface dimension (e.g., surface_w), which is an invalid index.
    if center_surface_x < 0:
        center_surface_x = 0
    elif center_surface_x >= surface_w:
        center_surface_x = surface_w - 1

    if center_surface_y < 0:
        center_surface_y = 0
    elif center_surface_y >= surface_h:
        center_surface_y = surface_h - 1

    # --- Standard (non-fuzzy) hit check ---
    if hit_distance <= 0:
        # Check if the calculated pixel is within the surface's bounds.
        if not (
            0 <= center_surface_x < surface_w
            and 0 <= center_surface_y < surface_h
        ):
            return False

        # Read the alpha value from the cairo surface data buffer.
        data = surface.get_data()
        stride = surface.get_stride()
        pixel_offset = center_surface_y * stride + center_surface_x * 4
        alpha = data[pixel_offset + 3]  # BGRA format, alpha is 4th byte
        return alpha > 0

    # --- Fuzzy hit check ---
    else:
        # The hit_distance is in screen pixels, making it intuitive.
        radius_px = int(round(hit_distance))
        # Add a safety clamp to prevent a deadlock if a huge value is given
        radius_px = min(radius_px, 50)
        radius_sq = radius_px * radius_px

        data = surface.get_data()
        stride = surface.get_stride()

        # Iterate over a square bounding box around the center point.
        for dy in range(-radius_px, radius_px + 1):
            for dx in range(-radius_px, radius_px + 1):
                # Check if the point is inside the circular radius
                if dx * dx + dy * dy > radius_sq:
                    continue

                px = center_surface_x + dx
                py = center_surface_y + dy

                # Check if the sample point is within surface bounds
                if not (0 <= px < surface_w and 0 <= py < surface_h):
                    continue

                # Check pixel alpha
                offset = py * stride + px * 4
                alpha = data[offset + 3]
                if alpha > 0:
                    return True  # Found an opaque pixel, it's a hit.

        # If we checked the whole area and found nothing, it's a miss.
        return False
