import cairo
import numpy as np
import logging
from ..models.ops import Ops
from .producer import OpsProducer


def rasterize_horizontally(surface,
                           ymax,
                           pixels_per_mm=10,
                           raster_size_mm=0.1):
    """
    Generate an engraving path for a Cairo surface, focusing on horizontal
    movement.

    Args:
        surface: A Cairo surface containing a black and white image.
        pixels_per_mm: Resolution of the image in pixels per millimeter.
        raster_size_mm: Distance between horizontal engraving lines in
        millimeters.

    Returns:
        A Ops object containing the optimized engraving path.
    """
    surface_format = surface.get_format()
    if surface_format != cairo.FORMAT_ARGB32:
        raise ValueError("Unsupported Cairo surface format")

    # Convert surface to a NumPy array
    width = surface.get_width()
    height = surface.get_height()
    data = np.frombuffer(surface.get_data(), dtype=np.uint8)
    data = data.reshape((height, width, 4))

    # Extract BGRA channels
    blue = data[:, :, 0]  # Blue channel
    green = data[:, :, 1]  # Green channel
    red = data[:, :, 2]  # Red channel
    alpha = data[:, :, 3]  # Alpha channel

    # Convert to grayscale (weighted average of RGB channels)
    bw_image = 0.2989 * red + 0.5870 * green + 0.1140 * blue

    # Threshold to black and white
    bw_image = (bw_image < 128).astype(np.uint8)

    # Optionally handle transparency (e.g., treat fully transparent
    # pixels as white)
    bw_image[alpha == 0] = 0  # Set fully transparent pixels to white (0)

    # Find the bounding box of the occupied area
    occupied_rows = np.any(bw_image, axis=1)
    occupied_cols = np.any(bw_image, axis=0)

    if not np.any(occupied_rows) or not np.any(occupied_cols):
        return Ops()  # No occupied area, return an empty path

    y_min, y_max = np.where(occupied_rows)[0][[0, -1]]
    x_min, x_max = np.where(occupied_cols)[0][[0, -1]]

    # Calculate dimensions in millimeters
    pixels_per_mm_x, pixels_per_mm_y = pixels_per_mm

    # Convert bounding box to millimeters
    x_min_mm = x_min / pixels_per_mm_x
    y_min_mm = y_min / pixels_per_mm_y
    y_max_mm = y_max / pixels_per_mm_y

    ops = Ops()

    # Iterate over rows in millimeters (floating-point) within the bounding box
    y_mm = y_min_mm
    while y_mm <= y_max_mm:
        # Convert y_mm to pixel coordinates (floating-point)
        y_px = y_mm * pixels_per_mm_y

        # TODO: Re-enable Y interpolation between nearest rows
        y1 = int(round(y_px))  # Use nearest neighbor instead of interpolation
        # y1 = int(np.floor(y_px))
        # y2 = int(np.ceil(y_px))
        # if y2 >= height:
        #     y2 = height - 1

        # Blend the two rows if y1 != y2
        # if y1 == y2:
        row = bw_image[y1, x_min:x_max + 1]  # Directly use the nearest row
        # else:
        #     alpha_y = y_px - y1
        #     row = (1 - alpha_y) * bw_image[y1, x_min:x_max + 1] \
        #         + alpha_y * bw_image[y2, x_min:x_max + 1]
        #     row = (row > 0.5).astype(np.uint8)  # Threshold the blended row

        # Find the start and end of black segments in the current row
        black_segments = np.where(np.diff(
            np.hstack(([0], row, [0]))
        ))[0].reshape(-1, 2)
        for start, end in black_segments:
            if row[start] == 1:  # Only process black segments
                # Convert segment start/end to mm (original simple version)
                start_mm = x_min_mm + (start / pixels_per_mm_x)
                end_mm = x_min_mm + ((end - 1) / pixels_per_mm_x)

                # Move to the start of the black segment
                ops.move_to(start_mm, ymax-y_mm)
                # Draw a line to the end of the black segment
                ops.line_to(end_mm, ymax-y_mm)

        # Move to the next raster line
        y_mm += raster_size_mm

    return ops


class Rasterizer(OpsProducer):
    """
    Generates rastered movements (using only straight lines)
    across filled pixels in the surface.
    """
    def run(self, machine, laser, surface, pixels_per_mm):
        width = surface.get_width()
        height = surface.get_height()
        logging.debug(f"Rasterizer received surface: {width}x{height} pixels")
        logging.debug(f"Rasterizer received pixels_per_mm: {pixels_per_mm}")

        ymax = surface.get_height()/pixels_per_mm[1]
        return rasterize_horizontally(
            surface,
            ymax,  # y max for axis inversion
            pixels_per_mm,
            laser.spot_size_mm[1]
        )

    def can_scale(self) -> bool:
        return False
