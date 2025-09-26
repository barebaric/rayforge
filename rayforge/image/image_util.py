import warnings
from typing import Optional, Tuple, Dict, Any
import logging
import cairo
import numpy

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

logger = logging.getLogger(__name__)


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


def get_physical_size_mm(image: pyvips.Image) -> Tuple[float, float]:
    """
    Determines the physical size of a vips image in mm.
    """
    try:
        xres = image.get("xres")
        yres = image.get("yres")

        # pyvips can default to a resolution of 1 pixel/mm if no resolution
        # info is available when creating an image in memory. This is a very
        # low resolution (25.4 DPI) and is usually not the intended value.
        # We treat this specific case as "resolution not set" and fall back
        # to the more common default of 96 DPI.
        if xres == 1.0 and yres == 1.0:
            raise pyvips.Error(
                "Default resolution of 1.0 px/mm detected, using fallback."
            )

        width_mm = image.width / xres
        height_mm = image.height / yres
    except pyvips.Error:
        width_mm = image.width * (25.4 / 96.0)
        height_mm = image.height * (25.4 / 96.0)
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
