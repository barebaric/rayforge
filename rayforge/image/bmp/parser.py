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
_SUPPORTED_BPP = 1, 8, 24, 32


def parse_bmp(data: bytes) -> Optional[Tuple[bytes, int, int, float, float]]:
    """
    Parse a BMP file and extract image data and metadata.

    Supports uncompressed 1-bit, 8-bit, 24-bit, and 32-bit BMPs. Handles
    BITMAPINFOHEADER, BITMAPV5HEADER, and the older BITMAPCOREHEADER formats.

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

        dib_header_size = struct.unpack("<I", data[14:18])[0]

        if bits_per_pixel in (1, 8):
            rgba_buffer = _parse_paletted_data(
                data,
                width,
                height,
                pixel_data_start,
                is_top_down,
                dib_header_size,
                bits_per_pixel=bits_per_pixel,
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
        logger.error(f"Failed to parse BMP headers or data: {e}")
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
        # bfOffBits is the 4 bytes at offset 10
        (pixel_data_offset,) = struct.unpack("<I", data[10:14])
        logger.debug(f"Pixel data starts at offset {pixel_data_offset}")
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
    logger.debug(f"DIB header size = {dib_header_size}")

    if dib_header_size == _BITMAPINFOHEADER_SIZE:
        return _parse_info_header(data)
    elif dib_header_size == _BITMAPCOREHEADER_SIZE:
        return _parse_core_header(data)
    elif dib_header_size == _BITMAPV5HEADER_SIZE:
        return _parse_v5_header(data)
    else:
        logger.error(f"Unsupported DIB header size: {dib_header_size}")
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
        f"INFOHEADER width={width} height={height} bpp={bits_per_pixel} "
        f"compression={compression} is_top_down={is_top_down}"
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
        f"COREHEADER width={width} height={height} bpp={bits_per_pixel}"
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
        logger.error(f"Unsupported compression type: {compression}")
        return False

    if bits_per_pixel not in _SUPPORTED_BPP:
        logger.error(
            f"Unsupported bpp: {bits_per_pixel}. "
            f"Only {_SUPPORTED_BPP} are supported."
        )
        return False

    return True


def _parse_info_palette(
    data: bytes, palette_offset: int, num_colors: int
) -> Optional[List[Tuple[int, int, int, int]]]:
    """Parse a BMP palette with 4-byte RGBQUAD entries."""
    palette_size = num_colors * 4
    if len(data) < palette_offset + palette_size:
        logger.error(
            f"Palette bytes (RGBQUAD) not present at offset {palette_offset}."
        )
        return None

    palette = []
    for i in range(0, palette_size, 4):
        b, g, r, _ = data[palette_offset + i : palette_offset + i + 4]
        palette.append((r, g, b, 255))

    logger.debug(f"Palette entries (INFO) read for {num_colors} colors.")
    return palette


def _parse_core_palette(
    data: bytes, palette_offset: int, num_colors: int
) -> Optional[List[Tuple[int, int, int, int]]]:
    """Parse a BMP palette with 3-byte RGBTRIPLE entries."""
    palette_size = num_colors * 3
    if len(data) < palette_offset + palette_size:
        logger.error(
            f"Palette bytes (RGBTRIPLE) not present at "
            f"offset {palette_offset}."
        )
        return None

    palette = []
    for i in range(0, palette_size, 3):
        b, g, r = data[palette_offset + i : palette_offset + i + 3]
        palette.append((r, g, b, 255))  # Convert to RGBA

    logger.debug(f"Palette entries (CORE) read for {num_colors} colors.")
    return palette


def _parse_paletted_data(
    data: bytes,
    width: int,
    height: int,
    pixel_data_start: int,
    is_top_down: bool,
    dib_header_size: int,
    bits_per_pixel: int,
) -> Optional[bytearray]:
    """Helper for parsing 1-bit and 8-bit paletted data."""
    is_core_header = dib_header_size == _BITMAPCOREHEADER_SIZE
    palette_offset = 14 + dib_header_size
    max_colors = 2**bits_per_pixel

    if is_core_header:
        num_colors = max_colors
        palette = _parse_core_palette(data, palette_offset, num_colors)
    else:
        colors_used_offset = 14 + 32  # Offset of biClrUsed field
        colors_used = struct.unpack(
            "<I", data[colors_used_offset : colors_used_offset + 4]
        )[0]
        num_colors = colors_used if colors_used > 0 else max_colors
        palette = _parse_info_palette(data, palette_offset, num_colors)

    if palette is None:
        return None

    if bits_per_pixel == 1:
        row_bytes = (width + 7) // 8
        process_row_func = _process_1bit_row
    else:  # bits_per_pixel == 8
        row_bytes = width
        process_row_func = _process_8bit_row

    row_size_padded = (row_bytes + 3) & ~3
    logger.debug(
        f"bpp={bits_per_pixel} "
        f"row_bytes={row_bytes} "
        f"row_size_padded={row_size_padded}"
    )

    rgba_buffer = bytearray(width * height * 4)
    for y in range(height):
        row_offset = _get_row_offset(
            y, height, row_size_padded, pixel_data_start, is_top_down
        )

        slice_end = row_offset + row_size_padded
        if slice_end > len(data):
            logger.error(
                f"Row {y} slice end ({slice_end}) exceeds data length "
                f"({len(data)}). File is likely truncated."
            )
            return None

        row_data = data[row_offset:slice_end]
        process_row_func(row_data, width, y, palette, rgba_buffer)

    return rgba_buffer


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
        f"{bits_per_pixel}-bit image, row size padded={row_size_padded}"
    )

    rgba_buffer = bytearray(width * height * 4)
    for y in range(height):
        row_offset = _get_row_offset(
            y, height, row_size_padded, pixel_data_start, is_top_down
        )

        slice_end = row_offset + row_size_padded
        if slice_end > len(data):
            logger.error(
                f"Row {y} slice end ({slice_end}) exceeds data length "
                f"({len(data)}). File is likely truncated."
            )
            return None

        row_data = data[row_offset:slice_end]
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


def _process_8bit_row(
    row_data: bytes,
    width: int,
    y: int,
    palette: List[Tuple[int, int, int, int]],
    rgba_buffer: bytearray,
):
    """Process a single row of 8-bit paletted data."""
    dest_row_start = y * width * 4
    for x in range(width):
        palette_index = row_data[x]
        r, g, b, a = palette[palette_index]
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
