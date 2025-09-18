import pytest
import cairo
from pathlib import Path
from typing import cast

from rayforge.importer.svg.importer import SvgImporter, MM_PER_PX_FALLBACK
from rayforge.importer.svg.renderer import SVG_RENDERER
from rayforge.importer.shared.util import parse_length
from rayforge.core.workpiece import WorkPiece


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
def basic_workpiece(basic_svg_data: bytes) -> WorkPiece:
    # Use the importer to create the workpiece, so it gets its size set.
    importer = SvgImporter(basic_svg_data)
    doc_items = importer.get_doc_items(vector_config=None)
    # FIX: Add assertion to handle None and cast to correct type for linter
    assert doc_items
    return cast(WorkPiece, doc_items[0])


@pytest.fixture
def transparent_workpiece(transparent_svg_data: bytes) -> WorkPiece:
    # Use the importer to create the workpiece, so it gets its size set.
    importer = SvgImporter(transparent_svg_data)
    doc_items = importer.get_doc_items(vector_config=None)
    # FIX: Add assertion to handle None and cast to correct type for linter
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


class TestSvgRenderer:
    def test_parse_length(self):
        assert parse_length("100mm") == (100.0, "mm")
        assert parse_length("200") == (200.0, "px")

    def test_get_natural_size(self, basic_workpiece: WorkPiece):
        size = SVG_RENDERER.get_natural_size(basic_workpiece)
        # FIX: Check for None before subscripting to satisfy linter
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
        svg_data = b'<svg width="1000px" height="500px"></svg>'
        importer = SvgImporter(svg_data)
        doc_items = importer.get_doc_items(vector_config=None)
        assert doc_items
        workpiece = cast(WorkPiece, doc_items[0])

        # The workpiece.render_chunk will use its current size and the
        # px_per_mm factor to determine total pixels.
        # width_px = (1000 * fallback) * (1 / fallback) = 1000
        # height_px = (500 * fallback) * (1 / fallback) = 500

        chunks = list(
            workpiece.render_chunk(
                pixels_per_mm_x=(1 / MM_PER_PX_FALLBACK),  # effectively 96 DPI
                pixels_per_mm_y=(1 / MM_PER_PX_FALLBACK),
                max_chunk_width=400,
                max_chunk_height=300,
            )
        )
        assert len(chunks) == 6  # 3 cols (1000/400) x 2 rows (500/300)

    def test_edge_cases(self):
        empty_svg_data = b'<svg xmlns="http://www.w3.org/2000/svg"/>'
        importer = SvgImporter(empty_svg_data)
        # The importer should still produce a workpiece, even if it has no size
        doc_items = importer.get_doc_items(vector_config=None)
        assert doc_items
        workpiece = cast(WorkPiece, doc_items[0])

        # Now test the renderer with this sizeless workpiece
        assert SVG_RENDERER.get_natural_size(workpiece) is None
