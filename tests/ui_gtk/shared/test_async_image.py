"""Tests for the async remote image loader (no network access)."""

import hashlib
import time
from unittest.mock import MagicMock, patch

import pytest
import pyvips
from gi.repository import GLib

from rayforge.ui_gtk.shared import async_image
from rayforge.ui_gtk.shared.async_image import (
    AsyncImage,
    cache_path_for_url,
    fetch_image_bytes,
    texture_from_bytes,
)

pytestmark = pytest.mark.ui

URL = "https://example.com/shot.png"


def _png_bytes() -> bytes:
    return pyvips.Image.black(8, 8).write_to_buffer(".png")


def _pump_until(condition, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        while GLib.MainContext.default().iteration(False):
            pass
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def _mock_download(data: bytes):
    response = MagicMock()
    response.read.return_value = data
    mock_urlopen = MagicMock()
    mock_urlopen.return_value.__enter__.return_value = response
    return patch.object(async_image.urllib.request, "urlopen", mock_urlopen)


def test_cache_path_is_stable_and_url_specific():
    path = cache_path_for_url(URL)
    assert path == cache_path_for_url(URL)
    assert path.name == hashlib.sha256(URL.encode("utf-8")).hexdigest()
    assert path != cache_path_for_url("https://example.com/other.png")


def test_fetch_image_bytes_reads_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(async_image, "IMAGE_CACHE_DIR", tmp_path)
    cache_path_for_url(URL).write_bytes(b"cached")
    with patch.object(async_image.urllib.request, "urlopen") as urlopen:
        assert fetch_image_bytes(URL) == b"cached"
    urlopen.assert_not_called()


def test_fetch_image_bytes_downloads_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(async_image, "IMAGE_CACHE_DIR", tmp_path)
    with _mock_download(b"image-data"):
        assert fetch_image_bytes(URL) == b"image-data"
    assert cache_path_for_url(URL).read_bytes() == b"image-data"


def test_fetch_image_bytes_rejects_oversized(tmp_path, monkeypatch):
    monkeypatch.setattr(async_image, "IMAGE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(async_image, "MAX_IMAGE_BYTES", 4)
    with _mock_download(b"12345"):
        assert fetch_image_bytes(URL) is None
    assert not cache_path_for_url(URL).exists()


def test_fetch_image_bytes_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(async_image, "IMAGE_CACHE_DIR", tmp_path)
    response = MagicMock()
    response.read.return_value = b"fallback-data"
    context = MagicMock()
    context.__enter__.return_value = response
    fallback = "https://example.com/fallback.png"
    with patch.object(
        async_image.urllib.request,
        "urlopen",
        side_effect=[OSError("not found"), context],
    ):
        assert fetch_image_bytes(URL, fallback) == b"fallback-data"
    assert cache_path_for_url(fallback).read_bytes() == b"fallback-data"


def test_texture_from_bytes_decodes_png():
    texture = texture_from_bytes(_png_bytes())
    assert texture is not None
    assert texture.get_width() == 8
    assert texture.get_height() == 8


def test_texture_from_bytes_rejects_garbage():
    assert texture_from_bytes(b"not an image") is None


def test_async_image_keeps_placeholder_on_failure():
    with patch.object(async_image, "fetch_image_bytes", return_value=None):
        widget = AsyncImage(URL)
    assert _pump_until(lambda: not widget._spinner.get_spinning())
    assert widget.get_first_child() is widget._spinner


def test_async_image_click_calls_callback_with_texture():
    callback = MagicMock()
    with patch.object(
        async_image, "fetch_image_bytes", return_value=_png_bytes()
    ):
        widget = AsyncImage(URL, on_click=callback)
        assert _pump_until(lambda: widget._texture is not None)
        widget._on_clicked(None, 1, 0, 0)
    callback.assert_called_once_with(widget._texture)


def test_async_image_click_without_callback_is_noop():
    with patch.object(
        async_image, "fetch_image_bytes", return_value=_png_bytes()
    ):
        widget = AsyncImage(URL)
        assert _pump_until(lambda: widget._texture is not None)
        widget._on_clicked(None, 1, 0, 0)
