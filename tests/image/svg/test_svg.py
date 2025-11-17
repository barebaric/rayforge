import pytest
import cairo
from pathlib import Path
from typing import cast
from unittest.mock import Mock

from rayforge.image.svg.importer import SvgImporter
from rayforge.image.svg.renderer import SVG_RENDERER
from rayforge.image.svg.svgutil import MM_PER_PX
from rayforge.image.util import parse_length
from rayforge.core.workpiece import WorkPiece
from rayforge.core.vectorization_spec import TraceSpec
from rayforge.core.geo import (
    Geometry,
    MoveToCommand,
    LineToCommand,
)
from rayforge.core.matrix import Matrix
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec


def _setup_workpiece_with_context(
    importer: SvgImporter, vectorization_spec=None
) -> WorkPiece:
    """Helper to run importer and correctly link workpiece to its source."""
    payload = importer.get_doc_items(vectorization_spec=vectorization_spec)
    assert payload is not None and payload.items
    source = payload.source
    wp = cast(WorkPiece, payload.items[0])

    mock_doc = Mock()
    mock_doc.source_assets = {source.uid: source}
    mock_doc.get_source_asset_by_uid.side_effect = mock_doc.source_assets.get

    mock_parent = Mock()
    mock_parent.doc = mock_doc
    mock_parent.get_world_transform.return_value = Matrix.identity()
    wp.parent = mock_parent

    return wp


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
def svg_with_offset_text_data() -> bytes:
    """
    SVG specifically for testing the vector misalignment bug.
    - The vector content (a rect) is in the top-left quadrant.
    - The text content is in the bottom-right.
    - This forces the trimmed bounding box (including text) to be much larger
      than the vector-only bounding box.
    """
    return b"""<svg xmlns="http://www.w3.org/2000/svg"
                width="100mm" height="100mm" viewBox="0 0 100 100">
                <rect x="10" y="10" width="40" height="40" fill="purple"/>
                <text x="90" y="90" font-family="sans-serif" font-size="10"
                      text-anchor="end">Test</text>
              </svg>"""


@pytest.fixture
def basic_workpiece(basic_svg_data: bytes) -> WorkPiece:
    importer = SvgImporter(basic_svg_data, source_file=Path("basic.svg"))
    return _setup_workpiece_with_context(importer)


@pytest.fixture
def transparent_workpiece(transparent_svg_data: bytes) -> WorkPiece:
    importer = SvgImporter(
        transparent_svg_data, source_file=Path("transparent.svg")
    )
    return _setup_workpiece_with_context(importer)


class TestSvgImporter:
    def test_importer_creates_workpiece_with_mm_size(
        self, basic_svg_data: bytes
    ):
        importer = SvgImporter(basic_svg_data, source_file=Path("test.svg"))
        payload = importer.get_doc_items(vectorization_spec=None)

        assert payload is not None
        wp = cast(WorkPiece, payload.items[0])
        assert isinstance(wp, WorkPiece)
        # The importer should set the size from the SVG's 'mm' dimensions.
        assert wp.size == pytest.approx((100.0, 50.0))

    def test_importer_sets_default_size_from_px(
        self, transparent_svg_data: bytes
    ):
        importer = SvgImporter(
            transparent_svg_data, source_file=Path("test.svg")
        )
        payload = importer.get_doc_items(vectorization_spec=None)

        assert payload is not None
        wp = cast(WorkPiece, payload.items[0])
        # The SVG content is 100px wide. The importer should convert this
        # to mm using the fallback DPI.
        expected_size_mm = 100.0 * MM_PER_PX
        assert wp.size == pytest.approx((expected_size_mm, expected_size_mm))

    def test_direct_vector_import_geometry(self, square_svg_data: bytes):
        """
        Tests the direct vector import path (vectorization_spec=None) for
        geometry extraction and transformation.
        """
        importer = SvgImporter(square_svg_data, source_file=Path("square.svg"))
        payload = importer.get_doc_items(vectorization_spec=None)

        assert payload is not None
        wp = cast(WorkPiece, payload.items[0])
        assert isinstance(wp, WorkPiece)

        # Verify workpiece size based on SVG width/height, considering content
        # trimming. The rect (10,10) to (90,90) in a 100x100 viewBox means
        # content is 80x80 units. This content is scaled to fill the
        # calculated trimmed 80x80mm workpiece.
        assert wp.size == pytest.approx((80.0, 80.0), 5)

        # Check if vectors were successfully imported
        assert wp.vectors is not None
        assert isinstance(wp.vectors, Geometry)

        # A simple rectangle, when converted to Path by svgelements and then
        # to Geometry in _get_doc_items_direct, should typically result in:
        # MOVE_TO, LINE_TO, LINE_TO, LINE_TO, CLOSE_PATH -> 5 commands.
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
        # The geometry must be normalized to a 1x1 unit.
        geo_rect_min_x, geo_rect_min_y, geo_rect_max_x, geo_rect_max_y = (
            wp.vectors.rect()
        )
        assert geo_rect_min_x == pytest.approx(0.0, abs=1e-3)
        assert geo_rect_min_y == pytest.approx(0.0, abs=1e-3)
        assert geo_rect_max_x == pytest.approx(1.0, abs=1e-3)
        assert geo_rect_max_y == pytest.approx(1.0, abs=1e-3)

    def test_direct_import_with_offset_text(
        self, svg_with_offset_text_data: bytes
    ):
        """
        Tests the fix for the vector misalignment bug caused by text.
        The workpiece size should be based on the trimmed bounds of ALL content
        (rect + text), but the vector geometry should be correctly placed
        within that larger frame.
        """
        importer = SvgImporter(
            svg_with_offset_text_data, source_file=Path("offset.svg")
        )
        wp = _setup_workpiece_with_context(importer)

        # 1. Check the final workpiece size.
        # The content (rect at 10,10 and text near 90,90) spans roughly 80%
        # of the 100mm canvas. The trim function should resize the workpiece
        # to this content. We expect a size of roughly 80x80mm.
        assert wp.size[0] == pytest.approx(80.0, abs=5)
        assert wp.size[1] == pytest.approx(80.0, abs=5)

        # 2. Check that the vectors are normalized correctly. The vector
        # content (the rect) should only occupy the left half of the
        # normalized space, because the text occupies the right half.
        assert wp.vectors is not None
        min_v, _, max_v, _ = wp.vectors.rect()
        assert min_v == pytest.approx(0.0, abs=1e-3)
        # The vector data should only span half the width (0.5) of the
        # full content area because the other half is text, which is ignored.
        assert max_v == pytest.approx(0.5, abs=0.05)

        # 3. CRITICAL: Check the final position and scale of the vectors.
        # The rect was at (10,10) with size (40,40) inside a viewbox that
        # gets trimmed to start at (10,10) and have size (80,80).
        # This means the rect should occupy the top-left quadrant of the
        # final 80mm x 80mm workpiece.
        # In a Y-up coordinate system (0,0 is bottom-left), the top-left
        # quadrant spans x=[0, 40] and y=[40, 80].
        bbox = wp.get_geometry_world_bbox()
        assert bbox is not None
        min_x, min_y, max_x, max_y = bbox
        # The trim process is raster-based and won't be perfectly aligned
        # with the vector geometry. We relax the tolerance to allow for this.
        assert min_x == pytest.approx(0.0, abs=1)
        assert min_y == pytest.approx(wp.size[1] / 2, abs=1)  # approx 40
        assert max_x == pytest.approx(wp.size[0] / 2, abs=1)  # approx 40
        assert max_y == pytest.approx(wp.size[1], abs=1)  # approx 80

    def test_traced_bitmap_import_geometry(self, transparent_svg_data: bytes):
        """
        Tests the traced bitmap import path (vectorization_spec provided).
        """
        importer = SvgImporter(
            transparent_svg_data, source_file=Path("trace.svg")
        )
        trace_spec = TraceSpec(threshold=0.5)

        payload = importer.get_doc_items(vectorization_spec=trace_spec)

        assert payload is not None
        wp = cast(WorkPiece, payload.items[0])
        assert isinstance(wp, WorkPiece)

        # The content is trimmed to 100x100px, which is converted to mm.
        expected_content_size_mm = 100.0 * MM_PER_PX
        # Use a looser tolerance to account for render/trace variance.
        assert wp.size[0] == pytest.approx(expected_content_size_mm, rel=1e-2)
        assert wp.size[1] == pytest.approx(expected_content_size_mm, rel=1e-2)

        # Check if vectors were generated through tracing
        assert wp.vectors is not None
        assert isinstance(wp.vectors, Geometry)
        assert len(wp.vectors.commands) > 4

        # Check the overall bounds of the TRACED geometry.
        # Per the new architecture, the vectors MUST be normalized to a
        # 1x1 box.
        geo_rect_min_x, geo_rect_min_y, geo_rect_max_x, geo_rect_max_y = (
            wp.vectors.rect()
        )
        assert geo_rect_min_x == pytest.approx(0.0, abs=1e-3)
        assert geo_rect_min_y == pytest.approx(0.0, abs=1e-3)
        assert geo_rect_max_x == pytest.approx(1.0, abs=1e-3)
        assert geo_rect_max_y == pytest.approx(1.0, abs=1e-3)


class TestSvgRenderer:
    def test_parse_length(self):
        assert parse_length("100mm") == (100.0, "mm")
        assert parse_length("200") == (200.0, "px")

    def test_get_natural_size(self, basic_workpiece: WorkPiece):
        size = SVG_RENDERER.get_natural_size(basic_workpiece)
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
        svg_data = (
            b'<svg width="1000px" height="500px" viewBox="0 0 1000 500"></svg>'
        )
        importer = SvgImporter(svg_data, source_file=Path("chunk.svg"))
        workpiece = _setup_workpiece_with_context(importer)

        expected_w_mm = 1000 * MM_PER_PX
        expected_h_mm = 500 * MM_PER_PX
        assert workpiece.size == pytest.approx((expected_w_mm, expected_h_mm))

        chunks = list(
            workpiece.render_chunk(
                pixels_per_mm_x=(1 / MM_PER_PX),  # effectively 96 DPI
                pixels_per_mm_y=(1 / MM_PER_PX),
                max_chunk_width=400,
                max_chunk_height=300,
            )
        )
        # 1000px width / 400px = 2.5 -> 3 chunks
        # 500px height / 300px = 1.66 -> 2 chunks
        assert len(chunks) == 6  # 3 cols x 2 rows

    def test_edge_cases(self):
        """
        Tests importer and renderer behavior with an empty/invalid SVG.
        """
        # 1. Test the importer
        empty_svg_data = b'<svg xmlns="http://www.w3.org/2000/svg"/>'
        importer = SvgImporter(empty_svg_data, source_file=Path("empty.svg"))

        # An empty SVG should not produce any importable items.
        payload = importer.get_doc_items()
        assert payload is None

        # 2. Test the renderer
        # Simulate a workpiece linked to a source with no metadata, which is
        # the state after trying to import an empty SVG.
        source = SourceAsset(
            source_file=Path("empty.svg"),
            original_data=empty_svg_data,
            renderer=SVG_RENDERER,
        )
        source.metadata = {}  # Empty metadata for an empty SVG

        workpiece = WorkPiece(name="empty_wp")
        # Manually create a basic generation_config to link to the source
        gen_config = SourceAssetSegment(
            source_asset_uid=source.uid,
            segment_mask_geometry=Geometry(),
            vectorization_spec=PassthroughSpec(),
        )
        workpiece.source_segment = gen_config

        # Set up mock parent structure so `workpiece.source` resolves correctly
        mock_doc = Mock()
        mock_doc.source_assets = {source.uid: source}
        mock_doc.get_source_asset_by_uid.side_effect = (
            mock_doc.source_assets.get
        )
        mock_parent = Mock()
        mock_parent.doc = mock_doc
        workpiece.parent = mock_parent

        # The renderer should not find a size for a workpiece with no metadata
        assert SVG_RENDERER.get_natural_size(workpiece) is None
