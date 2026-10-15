import io
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from rayforge.core.source_asset import SourceAsset
from rayforge.core.vectorization_spec import TraceSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image import import_file, importer_registry
from rayforge.image.base_importer import ImporterFeature
from rayforge.image.bitmap.importer import (
    GifImporter,
    TiffImporter,
    WebpImporter,
)
from rayforge.image.bitmap.renderer import BITMAP_RENDERER
from rayforge.image.registry import renderer_registry

WIDTH_PX = 120
HEIGHT_PX = 80


def _square_image(offset_x: int = 30, bands: int = 3) -> pyvips.Image:
    """A white image with a solid black square."""
    pixels = np.full((HEIGHT_PX, WIDTH_PX, bands), 255, dtype=np.uint8)
    pixels[20:60, offset_x : offset_x + 40, :] = 0
    image = pyvips.Image.new_from_array(pixels, interpretation="srgb")
    if bands == 1:
        image = image.copy(interpretation="b-w")
    return image


def _square_pixels(offset_x: int = 30) -> np.ndarray:
    pixels = np.full((HEIGHT_PX, WIDTH_PX), 255, dtype=np.uint8)
    pixels[20:60, offset_x : offset_x + 40] = 0
    return pixels


def _gif_bytes() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(_square_pixels()).save(buf, format="GIF")
    return buf.getvalue()


def _animated_gif_bytes() -> bytes:
    frames = [Image.fromarray(_square_pixels(x)) for x in (10, 70)]
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    return buf.getvalue()


def _tiff_bytes(dpi: float = 254.0, sixteen_bit: bool = False) -> bytes:
    image = _square_image(bands=1)
    if sixteen_bit:
        image = (image.cast("ushort") * 257).copy(interpretation="grey16")
    image = image.copy(xres=dpi / 25.4, yres=dpi / 25.4)
    return image.tiffsave_buffer(resunit="inch")


def _webp_bytes() -> bytes:
    return _square_image().webpsave_buffer(lossless=True)


FORMATS = [
    pytest.param(GifImporter, _gif_bytes, "logo.gif", id="gif"),
    pytest.param(TiffImporter, _tiff_bytes, "logo.tif", id="tiff"),
    pytest.param(WebpImporter, _webp_bytes, "logo.webp", id="webp"),
]


class TestRegistration:
    @pytest.mark.parametrize(
        "extension, importer_cls",
        [
            (".gif", GifImporter),
            (".tif", TiffImporter),
            (".tiff", TiffImporter),
            (".webp", WebpImporter),
        ],
    )
    def test_extension_lookup(self, extension, importer_cls):
        assert importer_registry.get_by_extension(extension) is importer_cls

    @pytest.mark.parametrize(
        "mime_type, importer_cls",
        [
            ("image/gif", GifImporter),
            ("image/tiff", TiffImporter),
            ("image/webp", WebpImporter),
        ],
    )
    def test_mime_lookup(self, mime_type, importer_cls):
        assert importer_registry.get_by_mime_type(mime_type) is importer_cls

    def test_renderer_is_registered(self):
        name = type(BITMAP_RENDERER).__name__
        assert renderer_registry.get(name) is BITMAP_RENDERER

    @pytest.mark.parametrize(
        "importer_cls", [GifImporter, TiffImporter, WebpImporter]
    )
    def test_features(self, importer_cls):
        assert importer_cls.features == {ImporterFeature.BITMAP_TRACING}


class TestImport:
    @pytest.mark.parametrize("importer_cls, make_data, name", FORMATS)
    def test_scan_reports_size(self, importer_cls, make_data, name):
        manifest = importer_cls(make_data(), Path(name)).scan()
        assert manifest.title == name
        assert manifest.errors == []
        assert manifest.natural_size_mm is not None
        width_mm, height_mm = manifest.natural_size_mm
        assert width_mm / height_mm == pytest.approx(WIDTH_PX / HEIGHT_PX)

    @pytest.mark.parametrize("importer_cls, make_data, name", FORMATS)
    def test_trace_creates_workpiece(self, importer_cls, make_data, name):
        data = make_data()
        result = importer_cls(data, Path(name)).get_doc_items(TraceSpec())
        assert result is not None and result.payload is not None
        source = result.payload.source
        assert isinstance(source, SourceAsset)
        assert source.renderer is BITMAP_RENDERER
        assert source.original_data == data
        assert (source.width_px, source.height_px) == (WIDTH_PX, HEIGHT_PX)
        assert source.metadata["_importer_class"] == importer_cls.__name__
        wp = cast(WorkPiece, result.payload.items[0])
        assert wp.boundaries is not None
        assert not wp.boundaries.is_empty()

    @pytest.mark.parametrize("importer_cls, make_data, name", FORMATS)
    def test_invalid_data_reports_error(self, importer_cls, make_data, name):
        importer = importer_cls(b"not an image", Path(name))
        manifest = importer.scan()
        assert manifest.errors
        result = importer_cls(b"not an image", Path(name)).get_doc_items(
            TraceSpec()
        )
        assert result is not None
        assert result.payload is None
        assert result.errors

    @pytest.mark.parametrize("importer_cls, make_data, name", FORMATS)
    def test_import_file_by_path(
        self, tmp_path, importer_cls, make_data, name
    ):
        path = tmp_path / name
        path.write_bytes(make_data())
        payload = import_file(path, vectorization_spec=TraceSpec())
        assert payload is not None
        assert payload.source.metadata["_importer_class"] == (
            importer_cls.__name__
        )

    def test_tiff_resolution_sets_physical_size(self):
        manifest = TiffImporter(_tiff_bytes(dpi=254.0), Path("a.tif")).scan()
        assert manifest.natural_size_mm == pytest.approx((12.0, 8.0))

    @pytest.mark.parametrize("importer_cls, make_data, name", FORMATS[::2])
    def test_missing_resolution_falls_back_to_96_dpi(
        self, importer_cls, make_data, name
    ):
        importer = importer_cls(make_data(), Path(name))
        expected = (WIDTH_PX * 25.4 / 96.0, HEIGHT_PX * 25.4 / 96.0)
        assert importer.scan().natural_size_mm == pytest.approx(expected)
        result = importer_cls(make_data(), Path(name)).get_doc_items(
            TraceSpec()
        )
        assert result is not None and result.payload is not None
        source = result.payload.source
        assert (source.width_mm, source.height_mm) == pytest.approx(expected)

    def test_sixteen_bit_tiff_is_traced(self):
        data = _tiff_bytes(sixteen_bit=True)
        result = TiffImporter(data, Path("deep.tif")).get_doc_items(
            TraceSpec()
        )
        assert result is not None and result.payload is not None
        wp = cast(WorkPiece, result.payload.items[0])
        assert wp.boundaries is not None
        assert not wp.boundaries.is_empty()

    def test_animated_gif_uses_first_frame(self):
        data = _animated_gif_bytes()
        result = GifImporter(data, Path("anim.gif")).get_doc_items(TraceSpec())
        assert result is not None and result.payload is not None
        source = result.payload.source
        assert (source.width_px, source.height_px) == (WIDTH_PX, HEIGHT_PX)


class TestRenderer:
    @pytest.mark.parametrize("importer_cls, make_data, name", FORMATS)
    def test_renders_base_image(self, importer_cls, make_data, name):
        image = BITMAP_RENDERER.render_base_image(make_data(), 0, 0)
        assert image is not None
        assert (image.width, image.height) == (WIDTH_PX, HEIGHT_PX)

    def test_renders_first_gif_frame_only(self):
        image = BITMAP_RENDERER.render_base_image(_animated_gif_bytes(), 0, 0)
        assert image is not None
        assert (image.width, image.height) == (WIDTH_PX, HEIGHT_PX)
        assert image.getpoint(20, 40)[0] < 50
        assert image.getpoint(90, 40)[0] > 200

    def test_invalid_data_renders_nothing(self):
        assert BITMAP_RENDERER.render_base_image(b"junk", 10, 10) is None
        assert BITMAP_RENDERER.render_base_image(b"", 10, 10) is None
