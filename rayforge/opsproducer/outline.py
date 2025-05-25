import cairo
import numpy as np
import cv2
from ..models.ops import Ops
from .producer import OpsProducer


def prepare_surface_for_tracing(surface):
    # Get the surface format
    surface_format = surface.get_format()

    # Determine the number of channels based on the format
    if surface_format == cairo.FORMAT_ARGB32:
        channels = 4  # ARGB or RGBA
        target_fmt = cv2.COLOR_BGRA2GRAY
    elif surface_format == cairo.FORMAT_RGB24:
        channels = 3  # RGB
        target_fmt = cv2.COLOR_BGR2GRAY
    else:
        raise ValueError("Unsupported Cairo surface format")

    # Make a copy of the image.
    width, height = surface.get_width(), surface.get_height()
    buf = surface.get_data()
    img = np.frombuffer(buf, dtype=np.uint8)
    img = img.reshape(height, width, channels).copy()

    # Replace transparent pixels with white
    if channels == 4:
        alpha = img[:, :, 3]  # Extract the alpha channel
        img[alpha == 0] = 255, 255, 255, 255

    # Convert to binary image (thresholding)
    return cv2.cvtColor(img, target_fmt)


def contours2ops(contours, pixels_per_mm, ymax):
    """
    The resulting Ops needs to be in machine coordinates, i.e. zero
    point must be at the bottom left, and units need to be mm.
    Since Cairo coordinates put the zero point at the top left, we must
    subtract Y from the machine's Y axis maximum.
    """
    ops = Ops()
    scale_x, scale_y = pixels_per_mm
    for contour in contours:
        # Smooth contour
        peri = cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, 0.00015*peri, True)

        # Append (scaled to mm)
        if len(contour) > 0:
            ops.move_to(contour[0][0][0]/scale_x,
                        ymax-contour[0][0][1]/scale_y)
            for point in contour:
                x, y = point[0]
                ops.line_to(x/scale_x, ymax-y/scale_y)
            ops.close_path()
    return ops


class OutlineTracer(OpsProducer):
    """
    Find external outlines for laser cutting.
    """
    def run(self, machine, laser, surface, pixels_per_mm):
        # Find contours of the black areas
        binary = prepare_surface_for_tracing(surface)
        binary = cv2.GaussianBlur(binary, (3, 3), 0)
        binary = cv2.adaptiveThreshold(binary, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        # Apply erosion to clean up edges
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(binary,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        ymax = surface.get_height()/pixels_per_mm[1]
        return contours2ops(contours, pixels_per_mm, ymax)


class EdgeTracer(OpsProducer):
    """
    Find all edges (including holes) for laser cutting.
    """
    def run(self, machine, laser, surface, pixels_per_mm):
        binary = prepare_surface_for_tracing(surface)
        binary = cv2.GaussianBlur(binary, (5, 5), 0)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8)
        )

        # Retrieve all contours (including holes)
        edges = cv2.Canny(binary, 10, 250)
        # Apply dilation to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges,
                                       cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)
        ymax = surface.get_height()/pixels_per_mm[1]
        return contours2ops(contours, pixels_per_mm, ymax)
