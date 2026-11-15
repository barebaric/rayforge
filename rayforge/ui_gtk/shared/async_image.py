"""Asynchronous loading of remote images with a disk cache.

Used by the addon detail view to show screenshots. Downloads happen
over plain HTTPS in a background thread, so no git installation is
required on any platform.
"""

import hashlib
import logging
import threading
import urllib.request
from pathlib import Path

import pyvips
from gi.repository import Gdk, GLib, Gtk

from ...config import CACHE_DIR
from .texture_loader import _rgba_bytes_to_texture

logger = logging.getLogger(__name__)

IMAGE_CACHE_DIR = CACHE_DIR / "addon-images"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1600
REQUEST_TIMEOUT = 30
USER_AGENT = "rayforge-addon-manager"


def cache_path_for_url(url: str) -> Path:
    """Returns the disk cache path for a remote image URL."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return IMAGE_CACHE_DIR / digest


def fetch_image_bytes(
    url: str, fallback_url: str | None = None
) -> bytes | None:
    """
    Downloads an image, trying a fallback URL when the first fails.

    The fallback covers screenshots pinned to a version tag that does
    not exist yet (for example development builds).
    """
    data = _fetch_image_bytes(url)
    if data is None and fallback_url:
        data = _fetch_image_bytes(fallback_url)
    return data


def _fetch_image_bytes(url: str) -> bytes | None:
    """
    Downloads an image, using and populating the disk cache.

    Returns None when the download fails or the image exceeds the
    size limit. Cached files are trusted and returned as-is.
    """
    path = cache_path_for_url(url)
    if path.is_file():
        try:
            return path.read_bytes()
        except OSError as e:
            logger.warning("Failed to read cached image %s: %s", url, e)

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT
        ) as response:
            data = response.read(MAX_IMAGE_BYTES + 1)
    except (OSError, ValueError) as e:
        logger.warning("Failed to download image %s: %s", url, e)
        return None

    if len(data) > MAX_IMAGE_BYTES:
        logger.warning("Image exceeds size limit, skipping: %s", url)
        return None

    try:
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError as e:
        logger.warning("Failed to cache image %s: %s", url, e)
    return data


def texture_from_bytes(data: bytes) -> Gdk.Texture | None:
    """
    Decodes image bytes into a Gdk.Texture.

    pyvips handles PNG, JPEG, and WebP alike and shrinks large
    screenshots on load, keeping memory use bounded.
    """
    try:
        image = pyvips.Image.thumbnail_buffer(
            data,
            MAX_IMAGE_DIMENSION,
            height=MAX_IMAGE_DIMENSION,
            size=pyvips.Size.DOWN,
        )
        image = image.colourspace("srgb")
        if not image.hasalpha():
            image = image.addalpha()
        image = image.cast("uchar")
        return _rgba_bytes_to_texture(
            image.write_to_memory(), image.width, image.height
        )
    except (pyvips.error.Error, OSError, ValueError) as e:
        logger.warning("Failed to decode image: %s", e)
        return None


class AsyncImage(Gtk.Box):
    """
    A widget that loads a remote image in the background.

    Shows a spinner placeholder until the image arrives, and keeps
    the placeholder when the download or decoding fails.
    """

    def __init__(
        self,
        url: str,
        min_height: int = 200,
        fallback_url: str | None = None,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self._url = url
        self._fallback_url = fallback_url
        self._destroyed = False
        self.set_size_request(-1, min_height)

        self._spinner = Gtk.Spinner()
        self._spinner.set_size_request(32, 32)
        self._spinner.set_halign(Gtk.Align.CENTER)
        self._spinner.set_valign(Gtk.Align.CENTER)
        self._spinner.set_vexpand(True)
        self._spinner.start()
        self.append(self._spinner)

        self._picture = Gtk.Picture()
        self._picture.set_can_shrink(True)
        self._picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        self._picture.set_vexpand(True)

        self.connect("destroy", self._on_destroy)
        thread = threading.Thread(target=self._worker, daemon=True)
        thread.start()

    def _on_destroy(self, _widget):
        self._destroyed = True

    def _worker(self):
        data = fetch_image_bytes(self._url, self._fallback_url)
        texture = texture_from_bytes(data) if data else None
        GLib.idle_add(self._apply, texture)

    def _apply(self, texture: Gdk.Texture | None) -> bool:
        if self._destroyed:
            return GLib.SOURCE_REMOVE
        self._spinner.stop()
        if texture is not None:
            self.remove(self._spinner)
            self._picture.set_paintable(texture)
            self.append(self._picture)
        return GLib.SOURCE_REMOVE
