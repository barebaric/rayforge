from pathlib import Path
import pytest
from rayforge.image import import_file
from rayforge.core.vectorization_spec import TraceSpec, PassthroughSpec
from rayforge.core.workpiece import WorkPiece


@pytest.fixture
def tests_root() -> Path:
    """Returns the path to the 'tests' directory."""
    return Path(__file__).parent.parent


class TestSegmentMaskGeometry:
    """Tests that segment_mask_geometry is properly populated during import."""

    def test_raster_import_populates_segment_mask_geometry(
        self, tests_root: Path
    ):
        """Test that raster import (PNG) populates segment_mask_geometry."""
        png_path = tests_root / "image/png/color.png"
        payload = import_file(png_path, vectorization_spec=TraceSpec())

        assert payload is not None
        assert payload.source is not None
        assert len(payload.items) == 1

        wp = payload.items[0]
        assert isinstance(wp, WorkPiece)
        assert wp.source_segment is not None
        assert wp.source_segment.pristine_geometry is not None
        assert not wp.source_segment.pristine_geometry.is_empty()

        # For raster imports, boundaries and segment_mask_geometry are
        # different (boundaries are Y-up, segment_mask_geometry is Y-down)
        # But both should be non-empty
        assert wp.boundaries is not None
        assert wp.source_segment.pristine_geometry is not None
        assert not wp.boundaries.is_empty()
        assert not wp.source_segment.pristine_geometry.is_empty()

    def test_vector_import_populates_segment_mask_geometry(
        self, tests_root: Path
    ):
        """Test that vector import (SVG) populates segment_mask_geometry."""
        svg_path = tests_root / "image/svg/nested-rect.svg"
        # Force merge strategy
        payload = import_file(
            svg_path,
            vectorization_spec=PassthroughSpec(create_new_layers=False),
        )

        assert payload is not None
        assert payload.source is not None
        assert len(payload.items) == 1

        wp = payload.items[0]
        assert isinstance(wp, WorkPiece)
        assert wp.source_segment is not None
        assert wp.source_segment.pristine_geometry is not None
        assert not wp.source_segment.pristine_geometry.is_empty()

        # The boundaries property returns a Y-up version, while the source
        # segment stores the original Y-down version. They should not be
        # identical.
        assert wp.boundaries is not None
        assert not wp.boundaries.is_empty()
        assert wp.boundaries.to_dict() != (
            wp.source_segment.pristine_geometry.to_dict()
        )

    def test_dxf_import_populates_segment_mask_geometry(
        self, tests_root: Path
    ):
        """Test that DXF import populates segment_mask_geometry."""
        dxf_path = tests_root / "image/dxf/circle.dxf"
        # Force merge strategy
        payload = import_file(
            dxf_path,
            vectorization_spec=PassthroughSpec(create_new_layers=False),
        )

        assert payload is not None
        assert payload.source is not None

        # DXF can create multiple workpieces, check at least one
        assert len(payload.items) >= 1

        # Check the first workpiece
        item = payload.items[0]
        # With merge strategy, we expect a WorkPiece directly
        assert isinstance(item, WorkPiece)
        wp = item
        assert wp.source_segment is not None
        assert wp.source_segment.pristine_geometry is not None

        # The boundaries property returns a Y-up version, while the source
        # segment stores the original Y-up version (since DXF is Y-up). They
        # should not be identical because of the normalization flip logic.
        assert wp.boundaries is not None
        assert not wp.boundaries.is_empty()
        assert wp.boundaries.to_dict() != (
            wp.source_segment.pristine_geometry.to_dict()
        )

    def test_segment_mask_geometry_is_always_geometry_object(
        self, tests_root: Path
    ):
        """
        Test that segment_mask_geometry is always a Geometry object,
        never None.
        """
        # Test with raster import
        png_path = tests_root / "image/png/color.png"
        raster_payload = import_file(png_path, vectorization_spec=TraceSpec())

        assert raster_payload is not None
        wp_raster = raster_payload.items[0]
        assert isinstance(wp_raster, WorkPiece)
        assert wp_raster.source_segment is not None
        assert wp_raster.source_segment.pristine_geometry is not None

        # Test with vector import
        svg_path = tests_root / "image/svg/nested-rect.svg"
        vector_payload = import_file(
            svg_path,
            vectorization_spec=PassthroughSpec(create_new_layers=False),
        )

        assert vector_payload is not None
        wp_vector = vector_payload.items[0]
        assert isinstance(wp_vector, WorkPiece)
        assert wp_vector.source_segment is not None
        assert wp_vector.source_segment.pristine_geometry is not None
