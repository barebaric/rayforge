"""
Retention policy of the in-memory image caches of workpieces and
source assets: each cache keeps at most a few images and a byte budget,
evicting the least recently used image first. The newest image is
always kept, even when it alone exceeds the budget.
"""

import pyvips

from ..shared.util.cache import LruCache

IMAGE_CACHE_MAX_ENTRIES = 4
IMAGE_CACHE_MAX_BYTES = 128 * 1024 * 1024

_VIPS_FORMAT_BYTES = {
    "uchar": 1,
    "char": 1,
    "ushort": 2,
    "short": 2,
    "uint": 4,
    "int": 4,
    "float": 4,
    "complex": 8,
    "double": 8,
    "dpcomplex": 16,
}


def image_nbytes(image: pyvips.Image) -> int:
    """The uncompressed size of the image's pixels in bytes."""
    sample = _VIPS_FORMAT_BYTES.get(image.format, 8)
    return image.width * image.height * image.bands * sample


def new_image_cache(
    max_entries: int = IMAGE_CACHE_MAX_ENTRIES,
) -> LruCache:
    """An LRU cache of images with the shared byte budget."""
    return LruCache(
        max_entries=max_entries,
        max_bytes=IMAGE_CACHE_MAX_BYTES,
        sizeof=image_nbytes,
    )
