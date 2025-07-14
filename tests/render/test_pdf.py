import pytest
import io
import cairo
from pypdf import PdfReader
from rayforge.render.pdf import PDFRenderer, parse_length, to_mm

# Helper functions to create sample PDFs using cairo
def create_basic_pdf():
    """Create a PDF with a 100pt x 50pt page containing a red rectangle."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 100, 50)
    cr = cairo.Context(surface)
    cr.set_source_rgb(1, 0, 0)  # Red
    cr.rectangle(0, 0, 100, 50)
    cr.fill()
    surface.finish()
    buf.seek(0)
    return buf.getvalue()

def create_transparent_pdf():
    """Create a PDF with a 200pt x 200pt page and a green rectangle from (50,50) to (150,150)."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 200, 200)
    cr = cairo.Context(surface)
    cr.set_source_rgb(0, 1, 0)  # Green
    cr.rectangle(50, 50, 100, 100)  # From (50,50) to (150,150)
    cr.fill()
    surface.finish()
    buf.seek(0)
    return buf.getvalue()

def create_large_pdf():
    """Create a large PDF with a 10000pt x 5000pt page filled with yellow."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 10000, 5000)
    cr = cairo.Context(surface)
    cr.set_source_rgb(1, 1, 0)  # Yellow
    cr.rectangle(0, 0, 10000, 5000)
    cr.fill()
    surface.finish()
    buf.seek(0)
    return buf.getvalue()

def create_empty_pdf():
    """Create an empty PDF page of 100pt x 100pt with no content."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 100, 100)
    # No drawing commands
    surface.finish()
    buf.seek(0)
    return buf.getvalue()

# Fixtures using the helper functions
@pytest.fixture
def basic_pdf():
    return create_basic_pdf()

@pytest.fixture
def transparent_pdf():
    return create_transparent_pdf()

@pytest.fixture
def large_pdf():
    return create_large_pdf()

@pytest.fixture
def empty_pdf():
    return create_empty_pdf()

# Test class for PDFRenderer
class TestPDFRenderer:
    ### Utility Function Tests
    def test_parse_length(self):
        """Test parsing length strings, defaulting to 'pt' for PDFs."""
        assert parse_length("100mm") == (100.0, "mm")
        assert parse_length("50.5cm") == (50.5, "cm")
        assert parse_length("200") == (200.0, "pt")  # PDF default unit is points
        assert parse_length("10in") == (10.0, "in")

    def test_to_mm_conversion(self):
        """Test conversion of various units to millimeters."""
        assert to_mm(72, "pt") == pytest.approx(25.4, abs=0.01)  # 72pt = 1 inch = 25.4mm
        assert to_mm(1, "in") == 25.4
        assert to_mm(10, "cm") == 100.0
        assert to_mm(5, "mm") == 5.0
        with pytest.raises(ValueError):
            to_mm(10, "px")  # Requires px_factor, not applicable for PDFs here

    ### Core Functionality Tests
    def test_get_natural_size(self, basic_pdf):
        """Test natural size in millimeters for a 100pt x 50pt PDF."""
        width_mm, height_mm = PDFRenderer.get_natural_size(basic_pdf)
        # Conversion: 1pt = 25.4/72 mm
        expected_width = 100 * 25.4 / 72  # ≈ 35.2778mm
        expected_height = 50 * 25.4 / 72   # ≈ 17.6389mm
        assert width_mm == pytest.approx(expected_width, abs=0.01)
        assert height_mm == pytest.approx(expected_height, abs=0.01)

    def test_get_aspect_ratio(self, basic_pdf):
        """Test aspect ratio calculation (width/height)."""
        ratio = PDFRenderer.get_aspect_ratio(basic_pdf)
        assert ratio == pytest.approx(2.0, abs=0.01)  # 100pt / 50pt = 2.0

    def test_render_workpiece(self, basic_pdf):
        """
        Test rendering to a Cairo surface at default pixels_per_mm (1 mm = 25px).
        1 mm = 1 pt * 25.4 / 72
        """
        surface = PDFRenderer.render_workpiece(basic_pdf)
        x_mm = 100 * 25.4 / 72
        y_mm = 50 * 25.4 / 72
        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == pytest.approx(x_mm * 25, abs=0.1)
        assert surface.get_height() == pytest.approx(y_mm * 25, abs=0.1)

    def test_get_margins(self, transparent_pdf):
        """Test margin calculation for a PDF with content in the center."""
        margins = PDFRenderer._get_margins(transparent_pdf)
        assert len(margins) == 4  # (left, top, right, bottom)
        # Page is 200pt x 200pt, content from (50,50) to (150,150)
        # Margins: 50pt / 200pt = 0.25 on all sides
        assert margins == pytest.approx((0.25, 0.25, 0.25, 0.25), abs=0.01)

    def test_prepare(self, transparent_pdf):
        """Test cropping the PDF to its content."""
        cropped_pdf = PDFRenderer.prepare(transparent_pdf)
        reader = PdfReader(io.BytesIO(cropped_pdf))
        page = reader.pages[0]
        media_box = page.mediabox
        # Original: [0,0,200,200], after cropping: [50,50,150,150]
        assert float(media_box.left) == pytest.approx(50, abs=1.0)
        assert float(media_box.bottom) == pytest.approx(50, abs=1.0)
        assert float(media_box.right) == pytest.approx(150, abs=1.0)
        assert float(media_box.top) == pytest.approx(150, abs=1.0)

    ### Chunk Rendering Tests
    def test_render_chunk(self, large_pdf):
        """Test chunk rendering without overlap."""
        chunks = list(PDFRenderer.render_chunk(
            large_pdf, width_px=10000, height_px=5000,
            chunk_width=3000, chunk_height=2000, overlap_x=0, overlap_y=0
        ))
        # Page: 10000pt x 5000pt, rendered to 10000px x 5000px (zoom=1.0)
        # Chunks: 4 cols (10000/3000) x 3 rows (5000/2000) = 12 chunks
        assert len(chunks) == 12
        for chunk, (x, y) in chunks:
            assert isinstance(chunk, cairo.ImageSurface)
            # Full chunks are 3000x2000, edge chunks are smaller
            if x < 9000:
                assert chunk.get_width() == 3000
            else:
                assert chunk.get_width() == 1000  # 10000 - 3*3000
            if y < 4000:
                assert chunk.get_height() == 2000
            else:
                assert chunk.get_height() == 1000  # 5000 - 2*2000

    def test_render_chunk_overlap(self, basic_pdf):
        """Test chunk rendering with overlap."""
        chunks = list(PDFRenderer.render_chunk(
            basic_pdf, width_px=100, height_px=50,
            chunk_width=40, chunk_height=30, overlap_x=2, overlap_y=2
        ))
        # Page: 100pt x 50pt, rendered to 100px x 50px
        # Chunks: 3 cols (100/40) x 2 rows (50/30) = 6 chunks
        assert len(chunks) == 6
        expected = [
            (0, 0, 42, 32),    # 40+2, 30+2
            (40, 0, 42, 32),
            (80, 0, 20, 32),   # 100-80=20
            (0, 30, 42, 20),   # 50-30=20
            (40, 30, 42, 20),
            (80, 30, 20, 20),
        ]
        for (chunk, (x, y)), (ex_x, ex_y, ex_w, ex_h) in zip(chunks, expected):
            assert x == ex_x
            assert y == ex_y
            assert chunk.get_width() == ex_w
            assert chunk.get_height() == ex_h

    def test_invalid_pdf_handling(self):
        """Test that invalid PDF data raises an exception."""
        invalid_pdf = b"not a PDF"
        with pytest.raises(Exception):
            PDFRenderer.render_workpiece(invalid_pdf)
