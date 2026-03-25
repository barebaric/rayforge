import pytest
import io
import cairo
from pathlib import Path
from rayforge.core.vectorization_spec import TraceSpec, PassthroughSpec
from rayforge.image.base_importer import ImporterFeature
from rayforge.image.pdf.pdf_trace import PdfTraceImporter


def create_pdf_with_shapes() -> bytes:
    """Create a PDF with shapes for tracing tests."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 100, 100)
    cr = cairo.Context(surface)

    # Draw filled shapes
    cr.set_source_rgb(0, 0, 0)
    cr.rectangle(10, 10, 30, 30)
    cr.fill()

    cr.arc(70, 70, 20, 0, 2 * 3.14159)
    cr.fill()

    surface.finish()
    buf.seek(0)
    return buf.getvalue()


def create_pdf_with_gradient() -> bytes:
    """Create a PDF with a gradient for tracing tests."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 100, 100)
    cr = cairo.Context(surface)

    pattern = cairo.LinearGradient(0, 0, 100, 100)
    pattern.add_color_stop_rgb(0, 0, 0, 0)
    pattern.add_color_stop_rgb(1, 1, 1, 1)
    cr.set_source(pattern)
    cr.rectangle(0, 0, 100, 100)
    cr.fill()

    surface.finish()
    buf.seek(0)
    return buf.getvalue()


def create_empty_pdf() -> bytes:
    """Create an empty PDF with no content."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 100, 100)
    surface.finish()
    buf.seek(0)
    return buf.getvalue()


def create_large_pdf() -> bytes:
    """Create a large PDF for resolution calculation tests."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 1000, 500)
    cr = cairo.Context(surface)
    cr.set_source_rgb(1, 0, 0)
    cr.rectangle(0, 0, 1000, 500)
    cr.fill()
    surface.finish()
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def pdf_with_shapes() -> bytes:
    return create_pdf_with_shapes()


@pytest.fixture
def pdf_with_gradient() -> bytes:
    return create_pdf_with_gradient()


@pytest.fixture
def empty_pdf() -> bytes:
    return create_empty_pdf()


@pytest.fixture
def large_pdf() -> bytes:
    return create_large_pdf()


class TestPdfTraceImporterScan:
    def test_scan_returns_manifest(self, pdf_with_shapes: bytes):
        importer = PdfTraceImporter(pdf_with_shapes, Path("test.pdf"))
        manifest = importer.scan()

        assert manifest.title == "test.pdf"
        assert manifest.natural_size_mm is not None
        assert len(manifest.errors) == 0

    def test_scan_returns_correct_dimensions(self, pdf_with_shapes: bytes):
        importer = PdfTraceImporter(pdf_with_shapes)
        manifest = importer.scan()

        expected_width = 100 * 25.4 / 72
        expected_height = 100 * 25.4 / 72
        assert manifest.natural_size_mm == pytest.approx(
            (expected_width, expected_height), rel=1e-3
        )

    def test_scan_handles_invalid_data(self):
        importer = PdfTraceImporter(b"not a pdf", Path("invalid.pdf"))
        manifest = importer.scan()

        assert len(manifest.errors) > 0


class TestPdfTraceImporterParse:
    def test_parse_returns_result(self, pdf_with_shapes: bytes):
        importer = PdfTraceImporter(pdf_with_shapes)
        result = importer.parse()

        assert result is not None
        assert len(result.layers) == 1
        assert result.layers[0].layer_id == "__default__"

    def test_parse_creates_rasterized_image(self, pdf_with_shapes: bytes):
        importer = PdfTraceImporter(pdf_with_shapes)
        result = importer.parse()

        assert result is not None
        assert importer._image is not None

    def test_parse_handles_invalid_data(self):
        importer = PdfTraceImporter(b"not a pdf")
        result = importer.parse()

        assert result is None
        assert len(importer._errors) > 0

    def test_parse_handles_empty_pdf(self, empty_pdf: bytes):
        importer = PdfTraceImporter(empty_pdf)
        result = importer.parse()

        assert result is not None
        assert importer._image is not None


class TestPdfTraceImporterVectorize:
    def test_vectorize_traces_shapes(self, pdf_with_shapes: bytes):
        importer = PdfTraceImporter(pdf_with_shapes)
        parse_result = importer.parse()
        assert parse_result is not None

        vectorize_result = importer.vectorize(parse_result, TraceSpec())
        assert vectorize_result is not None

        geo = vectorize_result.geometries_by_layer.get(None)
        assert geo is not None
        assert not geo.is_empty()

    def test_vectorize_only_accepts_trace_spec(self, pdf_with_shapes: bytes):
        importer = PdfTraceImporter(pdf_with_shapes)
        parse_result = importer.parse()
        assert parse_result is not None

        with pytest.raises(TypeError):
            importer.vectorize(parse_result, PassthroughSpec())


class TestPdfTraceImporterCreateSourceAsset:
    def test_create_source_asset(self, pdf_with_shapes: bytes):
        importer = PdfTraceImporter(pdf_with_shapes)
        parse_result = importer.parse()
        assert parse_result is not None

        source = importer.create_source_asset(parse_result)
        assert source is not None
        assert source.renderer is not None
        assert source.base_render_data is not None


class TestRenderResolutionCalculation:
    def test_calculate_render_resolution_basic(self):
        importer = PdfTraceImporter(b"")
        w, h = importer._calculate_render_resolution(100, 100)

        expected_w = int(100 * PdfTraceImporter._TRACE_PPM)
        expected_h = int(100 * PdfTraceImporter._TRACE_PPM)
        assert w == expected_w
        assert h == expected_h

    def test_calculate_render_resolution_respects_max_dim(self):
        importer = PdfTraceImporter(b"")
        large_mm = 1000
        w, h = importer._calculate_render_resolution(large_mm, large_mm)

        assert w <= PdfTraceImporter._MAX_RENDER_DIM
        assert h <= PdfTraceImporter._MAX_RENDER_DIM

    def test_calculate_render_resolution_returns_minimum_one(self):
        importer = PdfTraceImporter(b"")
        w, h = importer._calculate_render_resolution(0.001, 0.001)

        assert w >= 1
        assert h >= 1


class TestPdfTraceImporterFeatures:
    def test_has_bitmap_tracing_feature(self):
        assert ImporterFeature.BITMAP_TRACING in PdfTraceImporter.features

    def test_label(self):
        assert "Trace" in PdfTraceImporter.label
