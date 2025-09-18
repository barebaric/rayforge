import pytest
import io
import cairo
from pathlib import Path
from typing import cast

from rayforge.importer.pdf.importer import PdfImporter
from rayforge.importer.pdf.renderer import PDF_RENDERER
from rayforge.core.workpiece import WorkPiece


def create_pdf_data(width_pt: float, height_pt: float) -> bytes:
    """Helper to create sample PDF data of a given size in points."""
    buf = io.BytesIO()
    surface = cairo.PDFSurface(buf, width_pt, height_pt)
    cr = cairo.Context(surface)
    cr.set_source_rgb(1, 1, 0)  # Yellow
    cr.rectangle(0, 0, width_pt, height_pt)
    cr.fill()
    surface.finish()
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def basic_pdf_data() -> bytes:
    """PDF data for a 100pt x 50pt page."""
    return create_pdf_data(100, 50)


@pytest.fixture
def large_pdf_data() -> bytes:
    """PDF data for a 1000pt x 500pt page for chunking tests."""
    return create_pdf_data(1000, 500)


@pytest.fixture
def basic_workpiece(basic_pdf_data: bytes) -> WorkPiece:
    """A WorkPiece created from the basic PDF data, sized by the importer."""
    importer = PdfImporter(basic_pdf_data)
    doc_items = importer.get_doc_items(vector_config=None)
    assert doc_items
    return cast(WorkPiece, doc_items[0])


@pytest.fixture
def large_workpiece(large_pdf_data: bytes) -> WorkPiece:
    """A WorkPiece created from the large PDF data, sized by the importer."""
    importer = PdfImporter(large_pdf_data)
    doc_items = importer.get_doc_items(vector_config=None)
    assert doc_items
    return cast(WorkPiece, doc_items[0])


class TestPdfImporter:
    def test_importer_creates_workpiece_with_natural_size(
        self, basic_pdf_data: bytes
    ):
        """
        Tests the importer creates a WorkPiece with the correct initial size.
        """
        importer = PdfImporter(basic_pdf_data)
        doc_items = importer.get_doc_items(vector_config=None)

        assert doc_items
        wp = doc_items[0]
        assert isinstance(wp, WorkPiece)

        # The importer should set the size based on the PDF's natural
        # dimensions.
        # 1pt = 25.4/72 mm
        expected_width = 100 * 25.4 / 72
        expected_height = 50 * 25.4 / 72
        assert wp.size == pytest.approx((expected_width, expected_height))

    def test_importer_handles_invalid_data(self):
        """Tests the importer creates a WorkPiece even with invalid data."""
        importer = PdfImporter(b"this is not a pdf")
        doc_items = importer.get_doc_items(vector_config=None)
        assert doc_items is not None and len(doc_items) == 1
        assert isinstance(doc_items[0], WorkPiece)


class TestPdfRenderer:
    def test_get_natural_size(self, basic_workpiece: WorkPiece):
        """Test natural size calculation on the renderer."""
        size = PDF_RENDERER.get_natural_size(basic_workpiece)
        assert size is not None
        width_mm, height_mm = size

        expected_width = 100 * 25.4 / 72
        expected_height = 50 * 25.4 / 72
        assert width_mm == pytest.approx(expected_width)
        assert height_mm == pytest.approx(expected_height)

    def test_get_natural_aspect_ratio(self, basic_workpiece: WorkPiece):
        """Test aspect ratio on the workpiece, which uses the renderer."""
        ratio = basic_workpiece.get_natural_aspect_ratio()
        assert ratio is not None
        assert ratio == pytest.approx(2.0)

    def test_render_to_pixels(self, basic_workpiece: WorkPiece):
        """Test rendering to a Cairo surface."""
        surface = PDF_RENDERER.render_to_pixels(
            basic_workpiece, width=200, height=100
        )
        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 200
        assert surface.get_height() == 100

    def test_render_chunk(self, large_workpiece: WorkPiece):
        """Test chunk rendering without overlap."""
        # Set pixels_per_mm to achieve a 1-to-1 mapping from points to pixels
        px_per_mm = 72 / 25.4
        chunks = list(
            large_workpiece.render_chunk(
                pixels_per_mm_x=px_per_mm,
                pixels_per_mm_y=px_per_mm,
                max_chunk_width=300,
                max_chunk_height=200,
            )
        )
        # Expected total pixels: 1000x500
        # Chunks: 4 cols (ceil(1000/300)) x 3 rows (ceil(500/200)) = 12
        assert len(chunks) == 12

        chunk, (x, y) = chunks[-1]  # Last chunk
        assert x == 900
        assert y == 400
        assert chunk.get_width() == 100  # 1000 - 3*300
        assert chunk.get_height() == 100  # 500 - 2*200

    def test_renderer_handles_invalid_data_gracefully(self):
        """
        Test that the renderer does not raise exceptions for invalid data.
        """
        invalid_wp = WorkPiece(
            source_file=Path("invalid.pdf"),
            renderer=PDF_RENDERER,
            data=b"not a valid pdf",
        )

        assert PDF_RENDERER.get_natural_size(invalid_wp) is None
        assert PDF_RENDERER.render_to_pixels(invalid_wp, 100, 100) is None

        chunks = list(
            invalid_wp.render_chunk(
                pixels_per_mm_x=1,
                pixels_per_mm_y=1,
                max_chunk_width=100,
            )
        )
        assert len(chunks) == 0
