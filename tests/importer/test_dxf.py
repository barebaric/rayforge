import pytest
import io
import ezdxf
from typing import Optional
from rayforge.importer.dxf.importer import DxfImporter
from rayforge.core.workpiece import WorkPiece


# Fixtures
@pytest.fixture
def empty_dxf_importer():
    doc = ezdxf.new()  # type: ignore
    buffer = io.StringIO()
    doc.write(buffer)
    return DxfImporter(buffer.getvalue().encode("utf-8"))


@pytest.fixture
def line_dxf_importer():
    doc = ezdxf.new()  # type: ignore
    doc.header["$INSUNITS"] = 4  # Millimeters
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 50))
    buffer = io.StringIO()
    doc.write(buffer)
    return DxfImporter(buffer.getvalue().encode("utf-8"))


@pytest.fixture
def circle_dxf_importer():
    doc = ezdxf.new()  # type: ignore
    doc.header["$INSUNITS"] = 4  # Millimeters
    msp = doc.modelspace()
    msp.add_circle(center=(50, 50), radius=25)
    buffer = io.StringIO()
    doc.write(buffer)
    return DxfImporter(buffer.getvalue().encode("utf-8"))


@pytest.fixture
def inches_dxf_importer():
    doc = ezdxf.new()  # type: ignore
    doc.header["$INSUNITS"] = 1  # Inches
    msp = doc.modelspace()
    msp.add_line((0, 0), (1, 1))  # 1 inch line
    buffer = io.StringIO()
    doc.write(buffer)
    return DxfImporter(buffer.getvalue().encode("utf-8"))


# Utility
def get_first_workpiece(importer: DxfImporter) -> Optional[WorkPiece]:
    """
    Retrieves the first WorkPiece from the list of document items produced
    by the importer. Assumes the test DXF produces at least one item.
    """
    items = importer.get_doc_items()
    if not items:
        return None
    # For these tests, we expect a single WorkPiece.
    item = items[0]
    if isinstance(item, WorkPiece):
        return item
    return None


# Test cases
class TestDXFImporter:
    def test_empty_dxf(self, empty_dxf_importer):
        items = empty_dxf_importer.get_doc_items()
        assert items is not None and len(items) == 0

    def test_line_conversion(self, line_dxf_importer):
        wp = get_first_workpiece(line_dxf_importer)
        assert wp is not None
        # Bbox is (0,0) to (100,50). Importer normalizes to origin (0,0).
        assert wp.pos == pytest.approx((0.0, 0.0))
        assert wp.size == pytest.approx((100.0, 50.0))

    def test_circle_conversion(self, circle_dxf_importer):
        wp = get_first_workpiece(circle_dxf_importer)
        assert wp is not None

        # Bbox is (25,25) to (75,75). Importer normalizes the whole drawing
        # by (-25, -25), so the final workpiece position is at (0,0).
        assert wp.pos == pytest.approx((0.0, 0.0))
        assert wp.size == pytest.approx((50.0, 50.0))

    def test_unit_conversion(self, inches_dxf_importer):
        wp = get_first_workpiece(inches_dxf_importer)
        assert wp is not None
        assert wp.size == pytest.approx((25.4, 25.4))

    def test_get_natural_size(self, line_dxf_importer):
        wp = get_first_workpiece(line_dxf_importer)
        assert wp is not None
        size = wp.get_natural_size()
        assert size is not None
        assert size == pytest.approx((100.0, 50.0))

    def test_get_aspect_ratio(self, line_dxf_importer):
        wp = get_first_workpiece(line_dxf_importer)
        assert wp is not None
        ratio = wp.get_natural_aspect_ratio()
        assert ratio == pytest.approx(2.0)

    def test_render_to_pixels(self, line_dxf_importer):
        wp = get_first_workpiece(line_dxf_importer)
        assert wp is not None
        surface = wp.render_to_pixels(width=200, height=100)
        assert surface is not None
        assert surface.get_width() == 200
        assert surface.get_height() == 100

    def test_render_chunk_generator(self, line_dxf_importer):
        wp = get_first_workpiece(line_dxf_importer)
        assert wp is not None
        chunks = list(
            wp.render_chunk(
                pixels_per_mm_x=1,
                pixels_per_mm_y=1,
                max_chunk_width=40,
                max_chunk_height=30,
            )
        )
        assert len(chunks) == 6  # 3 cols x 2 rows

    def test_invalid_dxf_handling(self):
        invalid_dxf = b"invalid dxf content"
        importer = DxfImporter(invalid_dxf)
        items = importer.get_doc_items()
        assert items is not None and len(items) == 0
