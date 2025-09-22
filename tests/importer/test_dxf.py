import pytest
import io
import ezdxf
from typing import Optional
from unittest.mock import Mock

from rayforge.importer.dxf.importer import DxfImporter
from rayforge.core.workpiece import WorkPiece
from rayforge.core.matrix import Matrix


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


def _setup_workpiece_with_context(
    importer: DxfImporter,
) -> Optional[WorkPiece]:
    """
    Helper to run importer, correctly link workpiece to its source,
    and mock the document context for rendering tests.
    """
    payload = importer.get_doc_items(vector_config=None)
    if not payload or not payload.items:
        return None

    # For these tests, we expect a single WorkPiece.
    item = payload.items[0]
    if not isinstance(item, WorkPiece):
        return None
    wp = item

    source = payload.source

    mock_doc = Mock()
    mock_doc.import_sources = {source.uid: source}
    mock_doc.get_import_source_by_uid.side_effect = mock_doc.import_sources.get

    mock_parent = Mock()
    mock_parent.doc = mock_doc
    mock_parent.get_world_transform.return_value = Matrix.identity()
    wp.parent = mock_parent

    return wp


@pytest.fixture
def line_workpiece(line_dxf_importer) -> Optional[WorkPiece]:
    return _setup_workpiece_with_context(line_dxf_importer)


@pytest.fixture
def circle_workpiece(circle_dxf_importer) -> Optional[WorkPiece]:
    return _setup_workpiece_with_context(circle_dxf_importer)


@pytest.fixture
def inches_workpiece(inches_dxf_importer) -> Optional[WorkPiece]:
    return _setup_workpiece_with_context(inches_dxf_importer)


# Test cases
class TestDXFImporter:
    def test_empty_dxf(self, empty_dxf_importer):
        payload = empty_dxf_importer.get_doc_items(vector_config=None)
        assert payload is not None
        assert payload.source is not None
        assert len(payload.items) == 0

    def test_line_conversion(self, line_workpiece: WorkPiece):
        assert line_workpiece is not None
        # Bbox is (0,0) to (100,50). Importer normalizes to origin (0,0).
        assert line_workpiece.pos == pytest.approx((0.0, 0.0))
        assert line_workpiece.size == pytest.approx((100.0, 50.0))

    def test_circle_conversion(self, circle_workpiece: WorkPiece):
        assert circle_workpiece is not None

        # Bbox is (25,25) to (75,75). Importer normalizes the whole drawing
        # by (-25, -25), so the final workpiece position is at (0,0).
        assert circle_workpiece.pos == pytest.approx((0.0, 0.0))
        assert circle_workpiece.size == pytest.approx((50.0, 50.0))

    def test_unit_conversion(self, inches_workpiece: WorkPiece):
        assert inches_workpiece is not None
        assert inches_workpiece.size == pytest.approx((25.4, 25.4))

    def test_get_natural_size(self, line_workpiece: WorkPiece):
        assert line_workpiece is not None
        size = line_workpiece.get_natural_size()
        assert size is not None
        assert size == pytest.approx((100.0, 50.0))

    def test_get_aspect_ratio(self, line_workpiece: WorkPiece):
        assert line_workpiece is not None
        ratio = line_workpiece.get_natural_aspect_ratio()
        assert ratio == pytest.approx(2.0)

    def test_render_to_pixels(self, line_workpiece: WorkPiece):
        assert line_workpiece is not None
        surface = line_workpiece.render_to_pixels(width=200, height=100)
        assert surface is not None
        assert surface.get_width() == 200
        assert surface.get_height() == 100

    def test_render_chunk_generator(self, line_workpiece: WorkPiece):
        assert line_workpiece is not None
        chunks = list(
            line_workpiece.render_chunk(
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
        payload = importer.get_doc_items(vector_config=None)
        assert payload is None
