import pytest
import io
import cairo
from rayforge.importer.pdf import PDFRenderer


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


# Fixtures using the helper functions
@pytest.fixture
def basic_pdf_renderer():
    return PDFRenderer(create_basic_pdf())


@pytest.fixture
def large_pdf_renderer():
    """Create a large PDF for chunking tests."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, 1000, 500)  # Smaller than before for speed
    cr = cairo.Context(surface)
    cr.set_source_rgb(1, 1, 0)  # Yellow
    cr.rectangle(0, 0, 1000, 500)
    cr.fill()
    surface.finish()
    buf.seek(0)
    return PDFRenderer(buf.getvalue())


class TestPDFRenderer:
    def test_get_natural_size(self, basic_pdf_renderer):
        """Test natural size in millimeters for a 100pt x 50pt PDF."""
        width_mm, height_mm = basic_pdf_renderer.get_natural_size()
        # Conversion: 1pt = 25.4/72 mm
        expected_width = 100 * 25.4 / 72  # ≈ 35.2778mm
        expected_height = 50 * 25.4 / 72  # ≈ 17.6389mm
        assert width_mm == pytest.approx(expected_width, abs=0.01)
        assert height_mm == pytest.approx(expected_height, abs=0.01)

    def test_get_aspect_ratio(self, basic_pdf_renderer):
        """Test aspect ratio calculation (width/height)."""
        ratio = basic_pdf_renderer.get_aspect_ratio()
        assert ratio == pytest.approx(2.0, abs=0.01)  # 100pt / 50pt = 2.0

    def test_render_to_pixels(self, basic_pdf_renderer):
        """
        Test rendering to a Cairo surface at a specified size.
        """
        surface = basic_pdf_renderer.render_to_pixels(width=200, height=100)
        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 200
        assert surface.get_height() == 100

    def test_render_chunk(self, large_pdf_renderer):
        """Test chunk rendering without overlap."""
        chunks = list(
            large_pdf_renderer.render_chunk(
                width_px=1000,
                height_px=500,
                max_chunk_width=300,
                max_chunk_height=200,
                overlap_x=0,
                overlap_y=0,
            )
        )
        # Page: 1000px x 500px
        # Chunks: 4 cols (1000/300) x 3 rows (500/200) = 12 chunks
        assert len(chunks) == 12
        chunk, (x, y) = chunks[-1]  # Last chunk
        assert x == 900
        assert y == 400
        assert chunk.get_width() == 100  # 1000 - 3*300
        assert chunk.get_height() == 100  # 500 - 2*200

    def test_render_chunk_overlap(self, basic_pdf_renderer):
        """Test chunk rendering with overlap."""
        chunks = list(
            basic_pdf_renderer.render_chunk(
                width_px=100,
                height_px=50,
                max_chunk_width=40,
                max_chunk_height=30,
                overlap_x=2,
                overlap_y=2,
            )
        )
        # Page: 100px x 50px
        # Chunks: 3 cols (100/40) x 2 rows (50/30) = 6 chunks
        assert len(chunks) == 6

        last_chunk, (x, y) = chunks[-1]
        assert x == 80
        assert y == 30
        assert last_chunk.get_width() == 20  # 100-80
        assert last_chunk.get_height() == 20  # 50-30

    def test_invalid_pdf_handling(self):
        """
        Test that invalid PDF data is handled gracefully and does not raise
        an exception, but instead returns None or empty results.
        """
        invalid_pdf_data = b"this is not a valid pdf"
        renderer = PDFRenderer(invalid_pdf_data)

        # get_natural_size should return (None, None) and not raise.
        assert renderer.get_natural_size() == (None, None)

        # Rendering methods should return None and not raise.
        assert renderer.render_to_pixels(100, 100) is None

        # The chunk renderer should be an empty generator.
        chunks = list(renderer.render_chunk(100, 100, max_chunk_width=100))
        assert len(chunks) == 0
