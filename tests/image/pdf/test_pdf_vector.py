import pytest
import io
import cairo
from pathlib import Path
from rayforge.image.pdf.pdf_vector import PdfVectorImporter
from rayforge.core.vectorization_spec import PassthroughSpec


def create_pdf_with_line() -> bytes:
    """Create a PDF with a single line."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 100, 100)
    cr = cairo.Context(surface)
    cr.set_source_rgb(0, 0, 0)
    cr.move_to(10, 10)
    cr.line_to(90, 90)
    cr.stroke()
    surface.finish()
    buf.seek(0)
    return buf.getvalue()


def create_pdf_with_rectangle() -> bytes:
    """Create a PDF with a rectangle."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 100, 100)
    cr = cairo.Context(surface)
    cr.set_source_rgb(0, 0, 0)
    cr.rectangle(10, 10, 80, 80)
    cr.stroke()
    surface.finish()
    buf.seek(0)
    return buf.getvalue()


def create_pdf_with_bezier() -> bytes:
    """Create a PDF with a bezier curve."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 100, 100)
    cr = cairo.Context(surface)
    cr.set_source_rgb(0, 0, 0)
    cr.move_to(10, 50)
    cr.curve_to(30, 10, 70, 90, 90, 50)
    cr.stroke()
    surface.finish()
    buf.seek(0)
    return buf.getvalue()


def create_pdf_with_dashed_line() -> bytes:
    """Create a PDF with a dashed line."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 100, 100)
    cr = cairo.Context(surface)
    cr.set_source_rgb(0, 0, 0)
    cr.set_dash([10, 5], 0)
    cr.move_to(10, 50)
    cr.line_to(90, 50)
    cr.stroke()
    surface.finish()
    buf.seek(0)
    return buf.getvalue()


def create_pdf_with_dashed_bezier() -> bytes:
    """Create a PDF with a dashed bezier curve."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 100, 100)
    cr = cairo.Context(surface)
    cr.set_source_rgb(0, 0, 0)
    cr.set_dash([5, 3], 0)
    cr.move_to(10, 50)
    cr.curve_to(30, 10, 70, 90, 90, 50)
    cr.stroke()
    surface.finish()
    buf.seek(0)
    return buf.getvalue()


def create_pdf_with_circle() -> bytes:
    """Create a PDF with a circle (arcs)."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 100, 100)
    cr = cairo.Context(surface)
    cr.set_source_rgb(0, 0, 0)
    cr.arc(50, 50, 40, 0, 2 * 3.14159)
    cr.stroke()
    surface.finish()
    buf.seek(0)
    return buf.getvalue()


def create_empty_pdf() -> bytes:
    """Create an empty PDF with no drawings."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 100, 100)
    surface.finish()
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def pdf_with_line() -> bytes:
    return create_pdf_with_line()


@pytest.fixture
def pdf_with_rectangle() -> bytes:
    return create_pdf_with_rectangle()


@pytest.fixture
def pdf_with_bezier() -> bytes:
    return create_pdf_with_bezier()


@pytest.fixture
def pdf_with_dashed_line() -> bytes:
    return create_pdf_with_dashed_line()


@pytest.fixture
def pdf_with_dashed_bezier() -> bytes:
    return create_pdf_with_dashed_bezier()


@pytest.fixture
def pdf_with_circle() -> bytes:
    return create_pdf_with_circle()


@pytest.fixture
def empty_pdf() -> bytes:
    return create_empty_pdf()


class TestPdfVectorImporterScan:
    def test_scan_returns_manifest(self, pdf_with_line: bytes):
        importer = PdfVectorImporter(pdf_with_line, Path("test.pdf"))
        manifest = importer.scan()

        assert manifest.title == "test.pdf"
        assert manifest.natural_size_mm is not None
        assert len(manifest.errors) == 0

    def test_scan_returns_correct_dimensions(self, pdf_with_line: bytes):
        importer = PdfVectorImporter(pdf_with_line)
        manifest = importer.scan()

        expected_width = 100 * 25.4 / 72
        expected_height = 100 * 25.4 / 72
        assert manifest.natural_size_mm == pytest.approx(
            (expected_width, expected_height), rel=1e-3
        )

    def test_scan_handles_invalid_data(self):
        importer = PdfVectorImporter(b"not a pdf", Path("invalid.pdf"))
        manifest = importer.scan()

        assert len(manifest.errors) > 0


class TestPdfVectorImporterParse:
    def test_parse_returns_result(self, pdf_with_line: bytes):
        importer = PdfVectorImporter(pdf_with_line)
        result = importer.parse()

        assert result is not None
        assert result.document_bounds == (0.0, 0.0, 100.0, 100.0)
        assert len(result.layers) == 1

    def test_parse_handles_invalid_data(self):
        importer = PdfVectorImporter(b"not a pdf")
        result = importer.parse()

        assert result is None
        assert len(importer._errors) > 0


class TestPdfVectorImporterVectorize:
    def test_vectorize_extracts_line(self, pdf_with_line: bytes):
        importer = PdfVectorImporter(pdf_with_line)
        parse_result = importer.parse()
        assert parse_result is not None

        vectorize_result = importer.vectorize(parse_result, PassthroughSpec())
        assert vectorize_result is not None

        geo = vectorize_result.geometries_by_layer.get("__default__")
        assert geo is not None
        assert not geo.is_empty()

    def test_vectorize_extracts_rectangle(self, pdf_with_rectangle: bytes):
        importer = PdfVectorImporter(pdf_with_rectangle)
        parse_result = importer.parse()
        assert parse_result is not None

        vectorize_result = importer.vectorize(parse_result, PassthroughSpec())
        geo = vectorize_result.geometries_by_layer.get("__default__")
        assert geo is not None
        assert not geo.is_empty()

    def test_vectorize_extracts_bezier(self, pdf_with_bezier: bytes):
        importer = PdfVectorImporter(pdf_with_bezier)
        parse_result = importer.parse()
        assert parse_result is not None

        vectorize_result = importer.vectorize(parse_result, PassthroughSpec())
        geo = vectorize_result.geometries_by_layer.get("__default__")
        assert geo is not None
        assert not geo.is_empty()

    def test_vectorize_extracts_circle_as_beziers(
        self, pdf_with_circle: bytes
    ):
        importer = PdfVectorImporter(pdf_with_circle)
        parse_result = importer.parse()
        assert parse_result is not None

        vectorize_result = importer.vectorize(parse_result, PassthroughSpec())
        geo = vectorize_result.geometries_by_layer.get("__default__")
        assert geo is not None
        assert not geo.is_empty()

    def test_vectorize_empty_pdf_returns_empty_geometry(
        self, empty_pdf: bytes
    ):
        importer = PdfVectorImporter(empty_pdf)
        parse_result = importer.parse()
        assert parse_result is not None

        vectorize_result = importer.vectorize(parse_result, PassthroughSpec())
        geo = vectorize_result.geometries_by_layer.get("__default__")
        assert geo is not None
        assert geo.is_empty()


class TestDashExpansion:
    def test_dashed_line_creates_multiple_segments(
        self, pdf_with_dashed_line: bytes
    ):
        importer = PdfVectorImporter(pdf_with_dashed_line)
        parse_result = importer.parse()
        assert parse_result is not None

        vectorize_result = importer.vectorize(parse_result, PassthroughSpec())
        geo = vectorize_result.geometries_by_layer.get("__default__")
        assert geo is not None
        assert not geo.is_empty()

        segments = list(geo.segments())
        assert len(segments) > 1, "Dashed line should create multiple segments"

    def test_solid_line_creates_single_segment(self, pdf_with_line: bytes):
        importer = PdfVectorImporter(pdf_with_line)
        parse_result = importer.parse()
        assert parse_result is not None

        vectorize_result = importer.vectorize(parse_result, PassthroughSpec())
        geo = vectorize_result.geometries_by_layer.get("__default__")
        assert geo is not None

        segments = list(geo.segments())
        assert len(segments) == 1, "Solid line should create single segment"

    def test_dashed_bezier_creates_multiple_segments(
        self, pdf_with_dashed_bezier: bytes
    ):
        importer = PdfVectorImporter(pdf_with_dashed_bezier)
        parse_result = importer.parse()
        assert parse_result is not None

        vectorize_result = importer.vectorize(parse_result, PassthroughSpec())
        geo = vectorize_result.geometries_by_layer.get("__default__")
        assert geo is not None
        assert not geo.is_empty()

        segments = list(geo.segments())
        assert len(segments) > 1, (
            "Dashed bezier should create multiple segments"
        )


class TestDashPatternParsing:
    def test_parse_empty_dash_pattern(self):
        importer = PdfVectorImporter(b"")
        pattern, phase = importer._parse_dash_pattern("")
        assert pattern == []
        assert phase == 0.0

    def test_parse_simple_dash_pattern(self):
        importer = PdfVectorImporter(b"")
        pattern, phase = importer._parse_dash_pattern("[ 10 5 ] 0")
        assert pattern == [10.0, 5.0]
        assert phase == 0.0

    def test_parse_complex_dash_pattern(self):
        importer = PdfVectorImporter(b"")
        pattern, phase = importer._parse_dash_pattern(
            "[ 7.087 1.417 1.417 1.417 ] 0"
        )
        assert pattern == [7.087, 1.417, 1.417, 1.417]
        assert phase == 0.0

    def test_parse_dash_pattern_with_phase(self):
        importer = PdfVectorImporter(b"")
        pattern, phase = importer._parse_dash_pattern("[ 10 5 ] 3")
        assert pattern == [10.0, 5.0]
        assert phase == 3.0


class TestDashLineExpansion:
    def test_dash_line_returns_segments(self):
        importer = PdfVectorImporter(b"")
        segments = importer._dash_line(0, 0, 100, 0, [10, 5], 0, 0)

        assert len(segments) > 0
        for seg in segments:
            assert seg[0] in ("m", "l")

    def test_dash_line_respects_pattern(self):
        importer = PdfVectorImporter(b"")
        segments = importer._dash_line(0, 0, 100, 0, [10, 5], 0, 0)

        line_segments = [s for s in segments if s[0] == "l"]
        assert len(line_segments) > 1

    def test_dash_line_with_phase(self):
        importer = PdfVectorImporter(b"")
        segments_no_phase = importer._dash_line(0, 0, 100, 0, [10, 5], 0, 0)
        segments_with_phase = importer._dash_line(0, 0, 100, 0, [10, 5], 5, 0)

        line_no_phase = [s for s in segments_no_phase if s[0] == "l"]
        line_with_phase = [s for s in segments_with_phase if s[0] == "l"]
        first_seg_no_phase = line_no_phase[0]
        first_seg_with_phase = line_with_phase[0]
        len_no_phase = first_seg_no_phase[2].x - first_seg_no_phase[1].x
        len_with_phase = first_seg_with_phase[2].x - first_seg_with_phase[1].x
        assert len_no_phase != len_with_phase


class TestDashBezierExpansion:
    def test_dash_bezier_returns_segments(self):
        importer = PdfVectorImporter(b"")
        segments = importer._dash_bezier(
            0, 0, 25, -40, 75, 40, 100, 0, [10, 5], 0, 0
        )

        assert len(segments) > 0
        for seg in segments:
            assert seg[0] in ("m", "c")

    def test_dash_bezier_respects_pattern(self):
        importer = PdfVectorImporter(b"")
        segments = importer._dash_bezier(
            0, 0, 25, -40, 75, 40, 100, 0, [10, 5], 0, 0
        )

        curve_segments = [s for s in segments if s[0] == "c"]
        assert len(curve_segments) > 1


class TestBezierArcLength:
    def test_arc_length_positive(self):
        importer = PdfVectorImporter(b"")
        length = importer._bezier_arc_length(0, 0, 0, 100, 100, 100, 100, 0)
        assert length > 0

    def test_arc_length_straight_line(self):
        importer = PdfVectorImporter(b"")
        length = importer._bezier_arc_length(0, 0, 0, 0, 100, 0, 100, 0)
        assert length == pytest.approx(100, rel=0.1)


class TestBezierPoint:
    def test_bezier_point_start(self):
        importer = PdfVectorImporter(b"")
        px, py = importer._bezier_point(0, 0, 0, 25, -40, 75, 40, 100, 0)
        assert px == pytest.approx(0)
        assert py == pytest.approx(0)

    def test_bezier_point_end(self):
        importer = PdfVectorImporter(b"")
        px, py = importer._bezier_point(1, 0, 0, 25, -40, 75, 40, 100, 0)
        assert px == pytest.approx(100)
        assert py == pytest.approx(0)
