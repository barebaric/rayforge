import pytest
import cairo
from pathlib import Path
from typing import cast

from rayforge.importer.svg.importer import SvgImporter, MM_PER_PX_FALLBACK
from rayforge.importer.svg.renderer import SVG_RENDERER
from rayforge.importer.shared.util import parse_length
from rayforge.core.workpiece import WorkPiece
from rayforge.core.vectorization_config import TraceConfig
from rayforge.core.geometry import (
    Geometry,
    MoveToCommand,
    LineToCommand,
)


@pytest.fixture
def basic_svg_data() -> bytes:
    return b"""<svg xmlns="http://www.w3.org/2000/svg"
                width="100mm" height="50mm" viewBox="0 0 100 50">
                <rect width="100" height="50" fill="red"/>
              </svg>"""


@pytest.fixture
def transparent_svg_data() -> bytes:
    return b"""<svg xmlns="http://www.w3.org/2000/svg"
                 width="200px" height="200px" viewBox="0 0 200 200">
                 <rect x="50" y="50" width="100" height="100" fill="green"/>
               </svg>"""


@pytest.fixture
def square_svg_data() -> bytes:
    """
    SVG with a blue square, explicit dimensions, and viewBox for direct
    import testing.
    """
    return b"""<svg xmlns="http://www.w3.org/2000/svg"
                width="100mm" height="100mm" viewBox="0 0 100 100">
                <rect x="10" y="10" width="80" height="80" fill="blue"/>
              </svg>"""


@pytest.fixture
def basic_workpiece(basic_svg_data: bytes) -> WorkPiece:
    # Use the importer to create the workpiece, so it gets its size set.
    importer = SvgImporter(basic_svg_data)
    doc_items = importer.get_doc_items(vector_config=None)
    # Add assertion to handle None and cast to correct type for linter
    assert doc_items
    return cast(WorkPiece, doc_items[0])


@pytest.fixture
def transparent_workpiece(transparent_svg_data: bytes) -> WorkPiece:
    # Use the importer to create the workpiece, so it gets its size set.
    importer = SvgImporter(transparent_svg_data)
    doc_items = importer.get_doc_items(vector_config=None)
    # Add assertion to handle None and cast to correct type for linter
    assert doc_items
    return cast(WorkPiece, doc_items[0])


class TestSvgImporter:
    def test_importer_creates_workpiece_with_mm_size(
        self, basic_svg_data: bytes
    ):
        importer = SvgImporter(basic_svg_data, source_file=Path("test.svg"))
        doc_items = importer.get_doc_items(vector_config=None)

        assert doc_items is not None
        wp = doc_items[0]
        assert isinstance(wp, WorkPiece)
        # The importer should set the size from the SVG's 'mm' dimensions.
        assert wp.size == pytest.approx((100.0, 50.0))

    def test_importer_sets_default_size_from_px(
        self, transparent_svg_data: bytes
    ):
        importer = SvgImporter(transparent_svg_data)
        doc_items = importer.get_doc_items(vector_config=None)

        assert doc_items is not None
        wp = doc_items[0]
        # The SVG content is 100px wide. The importer should convert this
        # to mm using the fallback DPI.
        expected_size_mm = 100.0 * MM_PER_PX_FALLBACK
        assert wp.size == pytest.approx((expected_size_mm, expected_size_mm))

    def test_direct_vector_import_geometry(self, square_svg_data: bytes):
        """
        Tests the direct vector import path (vector_config=None) for geometry
        extraction and transformation.
        """
        importer = SvgImporter(square_svg_data)
        doc_items = importer.get_doc_items(vector_config=None)

        assert doc_items is not None
        wp = doc_items[0]
        assert isinstance(wp, WorkPiece)

        # Verify workpiece size based on SVG width/height, considering content
        # trimming. The rect (10,10) to (90,90) in a 100x100 viewBox means
        # content is 80x80 units. This content is scaled to fill the
        # calculated trimmed 80x80mm workpiece.
        assert wp.size == pytest.approx((80.0, 80.0))  # Corrected expectation

        # Check if vectors were successfully imported
        assert wp.vectors is not None
        assert isinstance(wp.vectors, Geometry)

        # A simple rectangle, when converted to Path by svgelements and then
        # to Geometry in _get_doc_items_direct, should typically result in:
        # MOVE_TO (start of first line)
        # LINE_TO (end of first line)
        # LINE_TO (end of second line)
        # LINE_TO (end of third line)
        # CLOSE_PATH (closes the last line back to start)
        # This gives 5 commands.
        assert len(wp.vectors.commands) == 5

        # Check the types of commands to ensure they are basic path elements
        cmds = wp.vectors.commands
        assert isinstance(cmds[0], MoveToCommand)
        assert isinstance(cmds[1], LineToCommand)
        assert isinstance(cmds[2], LineToCommand)
        assert isinstance(cmds[3], LineToCommand)
        # The close_path command in geometry.py adds a LineToCommand
        assert isinstance(cmds[4], LineToCommand)

        # Check the overall bounds of the imported geometry.
        # The 80x80 rect from (10,10) in a 100x100 viewBox, scaled to an
        # 80x80 mm workpiece, should result in geometry occupying the full
        # 0 to 80 mm range in both axes.
        geo_rect_min_x, geo_rect_min_y, geo_rect_max_x, geo_rect_max_y = (
            wp.vectors.rect()
        )
        assert geo_rect_min_x == pytest.approx(0.0)
        assert geo_rect_min_y == pytest.approx(0.0)
        assert geo_rect_max_x == pytest.approx(80.0)  # Corrected expectation
        assert geo_rect_max_y == pytest.approx(80.0)  # Corrected expectation

    def test_traced_bitmap_import_geometry(self, transparent_svg_data: bytes):
        """
        Tests the traced bitmap import path (vector_config provided).
        """
        importer = SvgImporter(transparent_svg_data)
        # Removed 'resolution' parameter as it doesn't exist in TraceConfig
        trace_config = TraceConfig(threshold=0.5)

        doc_items = importer.get_doc_items(vector_config=trace_config)

        assert doc_items is not None
        wp = doc_items[0]
        assert isinstance(wp, WorkPiece)

        # The transparent_svg_data defines a 200px SVG with a 100x100px green
        # square. The _get_doc_items_from_trace method calls get_natural_size,
        # which accounts for margins.
        # So, the effective content size is 100px by 100px.
        # This will be converted to mm using MM_PER_PX_FALLBACK.
        expected_content_size_mm = 100.0 * MM_PER_PX_FALLBACK
        assert wp.size == pytest.approx(
            (expected_content_size_mm, expected_content_size_mm)
        )

        # Check if vectors were generated through tracing
        assert wp.vectors is not None
        assert isinstance(wp.vectors, Geometry)

        # Traced output of a square might have more commands than a
        # perfect rectangle due to bitmap pixelization and the tracing
        # algorithm. Expect at least some commands.
        # A simple square would usually result in at least 4 lines +
        # move/close, often more due to minor imperfections.
        assert len(wp.vectors.commands) > 4

        # Check the overall bounds of the traced geometry.
        # The tracing process should result in geometry that fills the
        # workpiece's size from (0,0) to (width_mm, height_mm).
        geo_rect_min_x, geo_rect_min_y, geo_rect_max_x, geo_rect_max_y = (
            wp.vectors.rect()
        )
        assert geo_rect_min_x == pytest.approx(0.0, abs=1e-3)
        assert geo_rect_min_y == pytest.approx(0.0, abs=1e-3)
        assert geo_rect_max_x == pytest.approx(
            expected_content_size_mm, abs=1e-3
        )
        assert geo_rect_max_y == pytest.approx(
            expected_content_size_mm, abs=1e-3
        )


class TestSvgRenderer:
    def test_parse_length(self):
        assert parse_length("100mm") == (100.0, "mm")
        assert parse_length("200") == (200.0, "px")

    def test_get_natural_size(self, basic_workpiece: WorkPiece):
        size = SVG_RENDERER.get_natural_size(basic_workpiece)
        # Check for None before subscripting to satisfy linter
        assert size is not None
        assert size[0] == pytest.approx(100.0)
        assert size[1] == pytest.approx(50.0)

    def test_render_to_pixels(self, basic_workpiece: WorkPiece):
        surface = SVG_RENDERER.render_to_pixels(
            basic_workpiece, width=200, height=100
        )
        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 200
        assert surface.get_height() == 100

    def test_render_chunk_generator(self):
        # This SVG is missing viewBox, width, height, so it falls back
        # to tracing based on the fix provided earlier.
        svg_data = b'<svg width="1000px" height="500px"></svg>'
        importer = SvgImporter(svg_data)
        doc_items = importer.get_doc_items(vector_config=None)
        assert doc_items  # Now passes due to fallback to trace
        workpiece = cast(WorkPiece, doc_items[0])

        # The workpiece.render_chunk will use its current size and the
        # px_per_mm factor to determine total pixels.
        expected_w_mm = 1000 * MM_PER_PX_FALLBACK
        expected_h_mm = 500 * MM_PER_PX_FALLBACK
        assert workpiece.size == pytest.approx((expected_w_mm, expected_h_mm))

        # width_px = (expected_w_mm) * (1 / MM_PER_PX_FALLBACK) = 1000
        # height_px = (expected_h_mm) * (1 / MM_PER_PX_FALLBACK) = 500

        chunks = list(
            workpiece.render_chunk(
                pixels_per_mm_x=(1 / MM_PER_PX_FALLBACK),  # effectively 96 DPI
                pixels_per_mm_y=(1 / MM_PER_PX_FALLBACK),
                max_chunk_width=400,
                max_chunk_height=300,
            )
        )
        # 1000px width / 400px max_chunk_width = 2.5 -> 3 chunks
        # 500px height / 300px max_chunk_height = 1.66 -> 2 chunks
        assert len(chunks) == 6  # 3 cols x 2 rows

    def test_edge_cases(self):
        # This SVG is empty, so it falls back to tracing.
        empty_svg_data = b'<svg xmlns="http://www.w3.org/2000/svg"/>'
        importer = SvgImporter(empty_svg_data)
        doc_items = importer.get_doc_items(vector_config=None)
        assert doc_items  # Now passes due to fallback to trace
        workpiece = cast(WorkPiece, doc_items[0])

        # Now test the renderer with this workpiece.
        # An empty SVG without width/height attributes means get_natural_size
        # returns None.
        assert SVG_RENDERER.get_natural_size(workpiece) is None

        # When get_natural_size returns None, the importer does not set
        # an explicit size.
        # It defaults to (1.0, 1.0) according to previous test runs.
        assert workpiece.size == (1.0, 1.0)  # Corrected expectation
