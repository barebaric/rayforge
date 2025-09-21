import struct
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# Supported DIB header types
_BITMAPINFOHEADER_SIZE = 40
_BITMAPCOREHEADER_SIZE = 12
_BITMAPV5HEADER_SIZE = 124

# Supported compression types
_COMPRESSION_NONE = 0
_COMPRESSION_BITFIELDS = 3

# Supported bit depths
_SUPPORTED_BPP = (1, 24, 32)


def parse_bmp(data: bytes) -> Optional[Tuple[bytes, int, int, float, float]]:
    """
    Parse a BMP file and extract image data and metadata.

    Supports uncompressed 1-bit, 24-bit, and 32-bit BMPs. Handles both
    BITMAPINFOHEADER and the older BITMAPCOREHEADER formats.

    Args:
        data: Raw bytes of the BMP file.

    Returns:
        A tuple containing (RGBA pixel buffer, width, height, dpi_x, dpi_y)
        or None if parsing fails.
    """
    if not is_valid_bmp_signature(data):
        logger.error("Not a BMP file (missing 'BM' magic bytes).")
        return None

    try:
        pixel_data_start = parse_file_header(data)
        if pixel_data_start is None:
            return None

        header_info = parse_dib_header(data)
        if header_info is None:
            return None

        (
            width,
            height,
            bits_per_pixel,
            compression,
            dpi_x,
            dpi_y,
            is_top_down,
        ) = header_info

        if not _validate_format(bits_per_pixel, compression):
            return None

        if bits_per_pixel == 1:
            dib_header_size = struct.unpack("<I", data[14:18])[0]
            is_core = dib_header_size == _BITMAPCOREHEADER_SIZE
            rgba_buffer = _parse_1bit_data(
                data, width, height, pixel_data_start, is_top_down, is_core
            )
        else:
            rgba_buffer = _parse_rgb_data(
                data,
                width,
                height,
                bits_per_pixel,
                pixel_data_start,
                is_top_down,
            )

        if rgba_buffer is None:
            return None

        return bytes(rgba_buffer), width, height, dpi_x, dpi_y

    except (struct.error, IndexError, ValueError) as e:
        logger.error("Failed to parse BMP headers or data: %s", e)
        return None


def is_valid_bmp_signature(data: bytes) -> bool:
    """
    Check if the provided data starts with the BMP signature 'BM'.

    Args:
        data: The byte data of the file.

    Returns:
        True if the signature is valid, False otherwise.
    """
    return len(data) >= 2 and data[:2] == b"BM"


def parse_file_header(data: bytes) -> Optional[int]:
    """
    Parse the 14-byte BMP file header to find the pixel data offset.

    Args:
        data: The byte data of the BMP file.

    Returns:
        The integer offset where the pixel data begins, or None on failure.
    """
    if len(data) < 14:
        logger.error("Incomplete file header.")
        return None

    try:
        _, _, _, pixel_data_offset = struct.unpack("<IHHI", data[2:14])
        logger.debug("Pixel data starts at offset %d", pixel_data_offset)
        return pixel_data_offset
    except struct.error:
        logger.error("Failed to unpack file header.")
        return None


def parse_dib_header(
    data: bytes,
) -> Optional[Tuple[int, int, int, int, float, float, bool]]:
    """
    Parse the DIB (Device-Independent Bitmap) header.

    This function identifies and parses BITMAPINFOHEADER (40 bytes),
    BITMAPV5HEADER (124 bytes), or an older BITMAPCOREHEADER (12 bytes)
    to extract image metadata.

    Args:
        data: The byte data of the BMP file.

    Returns:
        A tuple containing
          (width, height, bpp, compression, dpi_x, dpi_y, is_top_down),
        or None on failure.
    """
    if len(data) < 18:
        logger.error("Incomplete DIB header size field.")
        return None

    dib_header_size = struct.unpack("<I", data[14:18])[0]
    logger.debug("DIB header size = %d", dib_header_size)

    if dib_header_size == _BITMAPINFOHEADER_SIZE:
        return _parse_info_header(data)
    elif dib_header_size == _BITMAPCOREHEADER_SIZE:
        return _parse_core_header(data)
    elif dib_header_size == _BITMAPV5HEADER_SIZE:
        return _parse_v5_header(data)
    else:
        logger.error("Unsupported DIB header size: %d", dib_header_size)
        return None


def _parse_info_header(
    data: bytes,
) -> Optional[Tuple[int, int, int, int, float, float, bool]]:
    """Parse a BITMAPINFOHEADER (40 bytes)."""
    if len(data) < 54:
        logger.error("Incomplete BITMAPINFOHEADER.")
        return None

    info = struct.unpack("<iihhiiiiii", data[18:54])
    raw_width, raw_height = info[0], info[1]
    width, height = raw_width, abs(raw_height)
    bits_per_pixel = info[3]
    compression = info[4]
    ppm_x, ppm_y = info[6], info[7]

    dpi_x = ppm_x * 0.0254 if ppm_x > 0 else 96.0
    dpi_y = ppm_y * 0.0254 if ppm_y > 0 else 96.0
    is_top_down = raw_height < 0

    logger.debug(
        "INFOHEADER width=%d height=%d bpp=%d compression=%d is_top_down=%s",
        width,
        height,
        bits_per_pixel,
        compression,
        is_top_down,
    )
    return (
        width,
        height,
        bits_per_pixel,
        compression,
        dpi_x,
        dpi_y,
        is_top_down,
    )


def _parse_v5_header(
    data: bytes,
) -> Optional[Tuple[int, int, int, int, float, float, bool]]:
    """
    Parse a BITMAPV5HEADER (124 bytes).

    The V5 header is a superset of the V4 and INFO headers. The first 40
    bytes are identical to BITMAPINFOHEADER, so we can reuse its parsing logic.
    """
    if len(data) < 138:  # 14 (file) + 124 (v5 header)
        logger.error("Incomplete BITMAPV5HEADER.")
        return None

    logger.debug("Parsing BITMAPV5HEADER by reusing INFOHEADER logic.")
    return _parse_info_header(data)


def _parse_core_header(
    data: bytes,
) -> Optional[Tuple[int, int, int, int, float, float, bool]]:
    """Parse a BITMAPCOREHEADER (12 bytes)."""
    if len(data) < 26:
        logger.error("Incomplete BITMAPCOREHEADER.")
        return None

    width, height, _, bits_per_pixel = struct.unpack("<HHHH", data[18:26])
    compression = _COMPRESSION_NONE
    dpi_x, dpi_y = 96.0, 96.0
    is_top_down = False

    logger.debug(
        "COREHEADER width=%d height=%d bpp=%d", width, height, bits_per_pixel
    )
    return (
        width,
        height,
        bits_per_pixel,
        compression,
        dpi_x,
        dpi_y,
        is_top_down,
    )


def _validate_format(bits_per_pixel: int, compression: int) -> bool:
    """Validate that the BMP format is supported."""
    # Allow BI_RGB (0) and BI_BITFIELDS (3), which is used for uncompressed
    # 32bpp images.
    if compression not in (_COMPRESSION_NONE, _COMPRESSION_BITFIELDS):
        logger.error("Unsupported compression type: %d", compression)
        return False

    if bits_per_pixel not in _SUPPORTED_BPP:
        logger.error(
            "Unsupported bpp: %d. Only %s are supported.",
            bits_per_pixel,
            _SUPPORTED_BPP,
        )
        return False

    return True


def _parse_1bit_data(
    data: bytes,
    width: int,
    height: int,
    pixel_data_start: int,
    is_top_down: bool,
    is_core_header: bool,
) -> Optional[bytearray]:
    """Parse 1-bit monochrome BMP data with a color palette."""
    logger.debug("Detected 1-bit BMP: %dx%d px", width, height)

    dib_header_size = (
        _BITMAPCOREHEADER_SIZE if is_core_header else _BITMAPINFOHEADER_SIZE
    )
    palette_offset = 14 + dib_header_size

    if is_core_header:
        palette = _parse_1bit_core_palette(data, palette_offset)
    else:
        palette = _parse_1bit_info_palette(data, palette_offset)

    if palette is None:
        return None

    row_bytes = (width + 7) // 8
    row_size_padded = (row_bytes + 3) & ~3
    logger.debug("row_bytes=%d row_size_padded=%d", row_bytes, row_size_padded)

    rgba_buffer = bytearray(width * height * 4)
    for y in range(height):
        row_offset = _get_row_offset(
            y, height, row_size_padded, pixel_data_start, is_top_down
        )
        if row_offset + row_bytes > len(data):
            logger.error(
                "Row %d start (%d) exceeds data length (%d).",
                y,
                row_offset,
                len(data),
            )
            return None
        row_data = data[row_offset : row_offset + row_size_padded]
        _process_1bit_row(row_data, width, y, palette, rgba_buffer)

    return rgba_buffer


def _parse_1bit_info_palette(
    data: bytes, palette_offset: int
) -> Optional[List[Tuple[int, int, int, int]]]:
    """Parse a 1-bit BMP palette with 4-byte RGBQUAD entries."""
    palette_size = 2 * 4
    if len(data) < palette_offset + palette_size:
        logger.error(
            "Palette bytes (RGBQUAD) not present at offset %d.", palette_offset
        )
        return None

    palette = []
    for i in range(0, palette_size, 4):
        b, g, r, _ = data[palette_offset + i : palette_offset + i + 4]
        palette.append((r, g, b, 255))

    logger.debug("Palette entries (INFO): %s", palette)
    return palette


def _parse_1bit_core_palette(
    data: bytes, palette_offset: int
) -> Optional[List[Tuple[int, int, int, int]]]:
    """Parse a 1-bit BMP palette with 3-byte RGBTRIPLE entries."""
    palette_size = 2 * 3
    if len(data) < palette_offset + palette_size:
        logger.error(
            "Palette bytes (RGBTRIPLE) not present at offset %d.",
            palette_offset,
        )
        return None

    palette = []
    for i in range(0, palette_size, 3):
        b, g, r = data[palette_offset + i : palette_offset + i + 3]
        palette.append((r, g, b, 255))  # Convert to RGBA

    logger.debug("Palette entries (CORE): %s", palette)
    return palette


def _parse_rgb_data(
    data: bytes,
    width: int,
    height: int,
    bits_per_pixel: int,
    pixel_data_start: int,
    is_top_down: bool,
) -> Optional[bytearray]:
    """Parse 24-bit or 32-bit RGB(A) BMP data."""
    bytes_per_pixel = bits_per_pixel // 8
    row_size_padded = (width * bytes_per_pixel + 3) & ~3
    logger.debug(
        "%d-bit image, row size padded=%d", bits_per_pixel, row_size_padded
    )

    rgba_buffer = bytearray(width * height * 4)
    for y in range(height):
        row_offset = _get_row_offset(
            y, height, row_size_padded, pixel_data_start, is_top_down
        )
        if row_offset + width * bytes_per_pixel > len(data):
            logger.error(
                "Row %d start (%d) exceeds data length (%d).",
                y,
                row_offset,
                len(data),
            )
            return None
        row_data = data[row_offset : row_offset + width * bytes_per_pixel]
        _process_rgb_row(row_data, width, bytes_per_pixel, y, rgba_buffer)

    return rgba_buffer


def _get_row_offset(
    y: int, height: int, row_size: int, data_start: int, is_top_down: bool
) -> int:
    """Calculate the byte offset for a specific row."""
    if is_top_down:
        return data_start + y * row_size
    else:
        return data_start + (height - 1 - y) * row_size


def _process_1bit_row(
    row_data: bytes,
    width: int,
    y: int,
    palette: List[Tuple[int, int, int, int]],
    rgba_buffer: bytearray,
):
    """Process a single row of 1-bit monochrome data."""
    dest_row_start = y * width * 4
    for x in range(width):
        byte_val = row_data[x // 8]
        bit = (byte_val >> (7 - (x % 8))) & 1
        r, g, b, a = palette[bit]
        rgba_buffer[dest_row_start + x * 4 : dest_row_start + x * 4 + 4] = (
            r,
            g,
            b,
            a,
        )


def _process_rgb_row(
    row_data: bytes,
    width: int,
    bytes_per_pixel: int,
    y: int,
    rgba_buffer: bytearray,
):
    """Process a single row of 24-bit or 32-bit RGB data."""
    dest_row_start = y * width * 4
    for x in range(width):
        src_idx = x * bytes_per_pixel
        dest_idx = dest_row_start + x * 4
        b, g, r = row_data[src_idx : src_idx + 3]
        a = row_data[src_idx + 3] if bytes_per_pixel == 4 else 255
        rgba_buffer[dest_idx : dest_idx + 4] = (r, g, b, a)
