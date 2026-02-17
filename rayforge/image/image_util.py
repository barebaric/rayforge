from typing import Optional, Tuple, Dict, Any
import logging
import cairo
import numpy
import pyvips
from ..core.geo import Geometry
from ..core.matrix import Matrix

logger = logging.getLogger(__name__)


def surface_to_grayscale(
    surface: cairo.ImageSurface,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Convert a Cairo ARGB32 surface to a grayscale array with alpha handling.

    Performs proper unpremultiplication of alpha and blends to white
    background for grayscale calculation.

    Args:
        surface: Cairo ImageSurface in FORMAT_ARGB32 format.

    Returns:
        Tuple of (grayscale_array, alpha_array) as numpy arrays.
        grayscale_array: uint8 array with values 0-255.
        alpha_array: float32 array with values 0.0-1.0.
    """
    width_px = surface.get_width()
    height_px = surface.get_height()
    stride = surface.get_stride()
    buf = surface.get_data()
    data_with_padding = numpy.ndarray(
        shape=(height_px, stride // 4, 4), dtype=numpy.uint8, buffer=buf
    )
    data = data_with_padding[:, :width_px, :]

    alpha = data[:, :, 3].astype(numpy.float32) / 255.0

    r = data[:, :, 2].astype(numpy.float32)
    g = data[:, :, 1].astype(numpy.float32)
    b = data[:, :, 0].astype(numpy.float32)

    alpha_safe = numpy.maximum(alpha, 1e-6)

    r_unpremult = r / alpha_safe
    g_unpremult = g / alpha_safe
    b_unpremult = b / alpha_safe

    r_unpremult = numpy.clip(r_unpremult, 0, 255)
    g_unpremult = numpy.clip(g_unpremult, 0, 255)
    b_unpremult = numpy.clip(b_unpremult, 0, 255)

    r_blended = 255.0 - (255.0 - r_unpremult) * alpha
    g_blended = 255.0 - (255.0 - g_unpremult) * alpha
    b_blended = 255.0 - (255.0 - b_unpremult) * alpha

    gray_image = (
        0.2989 * r_blended + 0.5870 * g_blended + 0.1140 * b_blended
    ).astype(numpy.uint8)

    return gray_image, alpha


def surface_to_binary(
    surface: cairo.ImageSurface,
    threshold: int = 128,
    invert: bool = False,
) -> numpy.ndarray:
    """
    Convert a Cairo ARGB32 surface to a binary array using thresholding.

    Converts the surface to grayscale and applies a threshold to produce
    a binary (0 or 1) output. Transparent pixels are always treated as
    white (0).

    Args:
        surface: Cairo ImageSurface in FORMAT_ARGB32 format.
        threshold: Brightness value (0-255) for binarization. Pixels with
            grayscale value below this threshold become black (1).
        invert: If True, invert the binarization logic. Pixels above the
            threshold become black (1).

    Returns:
        2D numpy array with values 0 (white/transparent) or 1 (black).

    Raises:
        ValueError: If the surface format is not ARGB32.
    """
    if surface.get_format() != cairo.FORMAT_ARGB32:
        raise ValueError("Unsupported Cairo surface format")

    width = surface.get_width()
    height = surface.get_height()
    data = numpy.frombuffer(surface.get_data(), dtype=numpy.uint8)
    data = data.reshape((height, width, 4))

    blue = data[:, :, 0]
    green = data[:, :, 1]
    red = data[:, :, 2]
    alpha = data[:, :, 3]

    grayscale = 0.2989 * red + 0.5870 * green + 0.1140 * blue

    if invert:
        binary = (grayscale > threshold).astype(numpy.uint8)
    else:
        binary = (grayscale < threshold).astype(numpy.uint8)

    binary[alpha == 0] = 0
    return binary


def convert_surface_to_grayscale_inplace(surface: cairo.ImageSurface) -> None:
    """
    Convert a Cairo ARGB32 surface to grayscale in place.

    Modifies the surface directly, converting RGB channels to grayscale
    while preserving the alpha channel.

    Args:
        surface: Cairo ImageSurface in FORMAT_ARGB32 format.

    Raises:
        ValueError: If the surface format is not ARGB32.
    """
    if surface.get_format() != cairo.FORMAT_ARGB32:
        raise ValueError("Unsupported Cairo surface format")

    width, height = surface.get_width(), surface.get_height()
    data = surface.get_data()
    data_array = numpy.frombuffer(data, dtype=numpy.uint8).reshape(
        (height, width, 4)
    )

    gray = (
        0.299 * data_array[:, :, 2]
        + 0.587 * data_array[:, :, 1]
        + 0.114 * data_array[:, :, 0]
    ).astype(numpy.uint8)

    data_array[:, :, :3] = gray[:, :, None]


def make_surface_transparent(
    surface: cairo.ImageSurface, threshold: int = 250
) -> None:
    """
    Make "almost white" pixels transparent in a Cairo ARGB32 surface.

    Modifies the surface in place. Pixels with average brightness above
    the threshold have their alpha channel set to 0.

    Args:
        surface: Cairo ImageSurface in FORMAT_ARGB32 format.
        threshold: Brightness threshold (0-255). Pixels with average
            RGB value >= threshold become transparent.

    Raises:
        ValueError: If the surface format is not ARGB32.
    """
    if surface.get_format() != cairo.FORMAT_ARGB32:
        raise ValueError("Surface must be in ARGB32 format.")

    width, height = surface.get_width(), surface.get_height()
    stride = surface.get_stride()

    data = surface.get_data()
    buf = numpy.frombuffer(data, dtype=numpy.uint8).reshape((height, stride))

    argb = buf.view(dtype=numpy.uint32)[:, :width]

    r = (argb >> 16) & 0xFF
    g = (argb >> 8) & 0xFF
    b = argb & 0xFF

    brightness = (
        r.astype(numpy.uint16)
        + g.astype(numpy.uint16)
        + b.astype(numpy.uint16)
    ) // 3
    mask = brightness >= threshold

    argb[mask] = (0x00 << 24) | (r[mask] << 16) | (g[mask] << 8) | b[mask]


def resize_and_crop_from_full_image(
    full_image: pyvips.Image,
    target_w: int,
    target_h: int,
    crop_window_px: Tuple[float, float, float, float],
) -> Optional[pyvips.Image]:
    """
    Scales a full source image up to a high resolution and then crops a
    window from it. This preserves maximum detail in the final cropped image.

    Args:
        full_image: The original, full-resolution pyvips image.
        target_w: The final desired width of the cropped image in pixels.
        target_h: The final desired height of the cropped image in pixels.
        crop_window_px: A tuple (x, y, w, h) defining the crop area in the
                        *original* full_image's pixel coordinates.

    Returns:
        The high-resolution cropped image, or None on failure.
    """
    crop_x, crop_y, crop_w, crop_h = map(int, crop_window_px)
    if (
        crop_w <= 0
        or crop_h <= 0
        or crop_x < 0
        or crop_y < 0
        or crop_x + crop_w > full_image.width
        or crop_y + crop_h > full_image.height
    ):
        return pyvips.Image.black(target_w, target_h, bands=4)

    # 1. Calculate scaling factors to determine how large the full image
    #    needs to be so that the cropped section matches the target size.
    scale_x = target_w / crop_w
    scale_y = target_h / crop_h

    # 2. Resize the entire source image to this new high resolution.
    # Check for and apply EXIF orientation tag if it exists.
    if full_image.get_typeof("orientation") != 0:
        try:
            full_image = full_image.autorot()
        except pyvips.Error:
            logger.warning("Failed to apply autorotate to image.")

    scaled_full_image = full_image.resize(scale_x, vscale=scale_y)

    # 3. Calculate the new crop window coordinates in the scaled image.
    scaled_crop_x = int(crop_x * scale_x)
    scaled_crop_y = int(crop_y * scale_y)

    # 4. Crop the final high-resolution window from the scaled full image.
    return safe_crop(
        scaled_full_image, scaled_crop_x, scaled_crop_y, target_w, target_h
    )


def safe_crop(
    image: pyvips.Image, x: int, y: int, w: int, h: int
) -> Optional[pyvips.Image]:
    """
    Crops a pyvips image, safely handling cases where the crop window is
    partially or completely outside the image bounds by calculating the
    intersection.

    Returns the cropped image, or None if the intersection is empty.
    """
    img_w, img_h = image.width, image.height
    # Calculate the intersection of the crop rectangle and the image bounds.
    final_x = max(0, x)
    final_y = max(0, y)
    end_x = min(x + w, img_w)
    end_y = min(y + h, img_h)
    final_w = max(0, end_x - final_x)
    final_h = max(0, end_y - final_y)

    if final_w > 0 and final_h > 0:
        return image.crop(final_x, final_y, final_w, final_h)

    return None


def extract_vips_metadata(image: pyvips.Image) -> Dict[str, Any]:
    """
    Extracts file-based and content-based metadata from a pyvips Image.
    """
    metadata = {
        "width": image.width,
        "height": image.height,
        "bands": image.bands,
        "format": image.format,
        "interpretation": str(image.interpretation),
    }
    all_fields = image.get_fields()
    for field in all_fields:
        if field in metadata:
            continue
        try:
            value = image.get(field)
            if isinstance(value, bytes):
                if "icc-profile" in field:
                    value = f"<ICC profile, {len(value)} bytes>"
                elif len(value) > 256:
                    value = f"<binary data, {len(value)} bytes>"
                else:
                    try:
                        # Attempt to decode using strict UTF-8.
                        value = value.decode("utf-8")
                    except UnicodeDecodeError:
                        # Fallback for non-decodable binary data.
                        value = f"<binary data, {len(value)} bytes>"
            elif not isinstance(
                value, (str, int, float, bool, list, dict, type(None))
            ):
                value = str(value)
            metadata[field] = value
        except Exception as e:
            logger.debug(f"Could not read metadata field '{field}': {e}")
    return metadata


def get_mm_per_pixel(image: pyvips.Image) -> Tuple[float, float]:
    """
    Determines mm per pixel from a vips image metadata. Falls back to 96 DPI.
    """
    try:
        # xres/yres are in pixels/mm
        xres = image.get("xres")
        yres = image.get("yres")

        # pyvips can default to a resolution of 1 pixel/mm if no resolution
        # info is available. This is a very low resolution (25.4 DPI) and is
        # usually not the intended value. We treat this specific case as
        # "resolution not set" and fall back to the more common 96 DPI.
        if xres == 1.0 and yres == 1.0:
            raise pyvips.Error(
                "Default resolution of 1.0 px/mm detected, using fallback."
            )

        # Invert to get mm/px
        return 1.0 / xres, 1.0 / yres
    except pyvips.Error:
        # fallback to 96 DPI
        mm_per_inch = 25.4
        dpi = 96.0
        return (mm_per_inch / dpi), (mm_per_inch / dpi)


def get_physical_size_mm(image: pyvips.Image) -> Tuple[float, float]:
    """
    Determines the physical size of a vips image in mm.
    """
    mm_per_px_x, mm_per_px_y = get_mm_per_pixel(image)
    width_mm = image.width * mm_per_px_x
    height_mm = image.height * mm_per_px_y
    return width_mm, height_mm


def normalize_to_rgba(image: pyvips.Image) -> Optional[pyvips.Image]:
    """
    Normalizes a pyvips image to a 4-band, 8-bit sRGB format (uchar RGBA).
    """
    try:
        if image.interpretation != "srgb":
            image = image.colourspace("srgb")
        if not image.hasalpha():
            image = image.addalpha()
        if image.bands != 4:
            logger.warning(
                f"Image normalization had {image.bands} bands, cropping to 4."
            )
            image = image[0:4]
        if image.format != "uchar":
            image = image.cast("uchar")
        return image if image.bands == 4 else None
    except pyvips.Error as e:
        logger.error(f"Failed to normalize image to RGBA: {e}")
        return None


def vips_rgba_to_cairo_surface(image: pyvips.Image) -> cairo.ImageSurface:
    """
    Converts a 4-band RGBA pyvips image to a Cairo ARGB32 ImageSurface.
    """
    assert image.bands == 4, "Input image must be normalized to RGBA first"
    assert image.format == "uchar", "Input image must be 8-bit uchar"

    # Premultiply alpha. This promotes the image format to float.
    premultiplied_float = image.premultiply()

    # Cast the image back to uchar (8-bit) after premultiplication.
    premultiplied_uchar = premultiplied_float.cast("uchar")

    # Get the raw RGBA pixel data from the correctly formatted image.
    rgba_memory = premultiplied_uchar.write_to_memory()

    # Use numpy for robust channel shuffling from RGBA to BGRA, which is
    # the format Cairo expects for ARGB32 surfaces.
    rgba_array = numpy.frombuffer(rgba_memory, dtype=numpy.uint8).reshape(
        [premultiplied_uchar.height, premultiplied_uchar.width, 4]
    )
    bgra_array = numpy.ascontiguousarray(rgba_array[..., [2, 1, 0, 3]])

    # Create the Cairo surface from the correctly ordered BGRA numpy array.
    data = memoryview(bgra_array)
    surface = cairo.ImageSurface.create_for_data(
        data,
        cairo.FORMAT_ARGB32,
        premultiplied_uchar.width,
        premultiplied_uchar.height,
    )
    return surface


def _render_geometry_to_vips_mask(
    geometry: Geometry, width: int, height: int
) -> pyvips.Image:
    """Renders a Geometry object to a single-band 8-bit vips mask image."""
    surface = cairo.ImageSurface(cairo.FORMAT_A8, width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.paint()

    # Draw the geometry filled with white
    ctx.set_source_rgba(1, 1, 1, 1)
    geometry.to_cairo(ctx)
    ctx.fill()

    # Handle Cairo stride padding (e.g. if width is not multiple of 4)
    stride = surface.get_stride()
    cairo_data = surface.get_data()

    if stride == width:
        return pyvips.Image.new_from_memory(
            cairo_data, width, height, 1, "uchar"
        )

    # Remove stride padding using numpy to prevent mask distortion
    arr = numpy.frombuffer(cairo_data, dtype=numpy.uint8).reshape(
        (height, stride)
    )
    clean_data = numpy.ascontiguousarray(arr[:, :width]).tobytes()

    return pyvips.Image.new_from_memory(clean_data, width, height, 1, "uchar")


def apply_mask_to_vips_image(
    full_image: pyvips.Image, mask_geo: Geometry
) -> Optional[pyvips.Image]:
    """
    Masks a vips image using a geometry mask, making areas outside the
    geometry transparent. Does NOT crop the image.

    Expects the mask_geo to be NORMALIZED to a 0-1 Y-DOWN coordinate space.
    """
    if mask_geo.is_empty():
        # If the mask is empty, we return the image as-is, which is the
        # expected behavior for unmasked "as-is" PDF imports.
        return full_image

    rgba_image = normalize_to_rgba(full_image)
    if not rgba_image:
        return None

    # Scale the normalized mask geometry to the image's pixel dimensions.
    scaled_mask = mask_geo.copy()
    scale_matrix = Matrix.scale(rgba_image.width, rgba_image.height)
    scaled_mask.transform(scale_matrix.to_4x4_numpy())

    mask_vips = _render_geometry_to_vips_mask(
        scaled_mask, rgba_image.width, rgba_image.height
    )

    # Intersect the mask with the original alpha channel.
    # mask_vips is 255 inside the geometry, 0 outside.
    # We want: FinalAlpha = OriginalAlpha if Mask else 0.
    original_alpha = rgba_image[3]
    final_alpha = (mask_vips > 128).ifthenelse(original_alpha, 0)

    # Return RGBA with the new intersected alpha
    return rgba_image[0:3].bandjoin(final_alpha)
