import cairo
import pytest
from pathlib import Path
from typing import Tuple, cast
from dataclasses import asdict
from blinker import Signal
from rayforge.core.doc import Doc
from rayforge.core.source_asset import SourceAsset
from rayforge.core.item import DocItem
from rayforge.core.matrix import Matrix
from rayforge.core.tab import Tab
from rayforge.core.geo import Geometry
from rayforge.core.workpiece import WorkPiece
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import LayerImportMode, PassthroughSpec
from rayforge.image.svg.renderer import SvgRenderer
from rayforge.image import import_file


@pytest.fixture
def sample_svg_data() -> bytes:
    """Provides a simple SVG with defined dimensions in mm."""
    svg = """
    <svg width="100mm" height="50mm" viewBox="0 0 100 50"
         xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="100" height="50" fill="blue"/>
    </svg>
    """
    return svg.encode("utf-8")


@pytest.fixture
def doc_with_workpiece(
    sample_svg_data: bytes, tmp_path: Path
) -> Tuple[Doc, WorkPiece, SourceAsset]:
    """
    Creates a Doc with a single WorkPiece linked to a SourceAsset,
    which is the correct way to test a WorkPiece's data-dependent methods.
    """
    # Use the real import function to handle the entire import process.
    svg_file = tmp_path / "test_rect.svg"
    svg_file.write_bytes(sample_svg_data)
    payload = import_file(
        svg_file,
        vectorization_spec=PassthroughSpec(
            layer_import_mode=LayerImportMode.FLATTEN
        ),
    )

    assert payload is not None
    source = payload.source
    wp = cast(WorkPiece, payload.items[0])

    doc = Doc()
    doc.add_asset(source)
    doc.active_layer.add_child(wp)
    return doc, wp, source


@pytest.fixture
def workpiece_instance(
    doc_with_workpiece: Tuple[Doc, WorkPiece, SourceAsset],
):
    """Provides the WorkPiece instance from the doc_with_workpiece fixture."""
    return doc_with_workpiece[1]


class TestWorkPiece:
    def test_initialization(self, workpiece_instance, sample_svg_data):
        wp = workpiece_instance
        assert wp.name == "test_rect"
        assert wp.source_file is not None
        assert wp.source_file.name == "test_rect.svg"
        # Renderer is accessed via the source asset now
        assert wp.source is not None
        assert isinstance(wp.source.renderer, SvgRenderer)
        # For a trimmed SVG, wp.data should return the base_render_data
        assert wp.data is not None
        assert sample_svg_data in wp.source.original_data

        # With padding, the position and size are slightly different from 0,0
        # and 100,50
        assert wp.pos[0] == pytest.approx(-1.0, abs=1e-4)
        assert wp.pos[1] == pytest.approx(-1.0, abs=1e-4)
        assert wp.size == pytest.approx((102.0, 52.0), abs=1e-4)

        assert wp.angle == pytest.approx(0.0)
        # Importer sets size, so matrix is not identity
        assert wp.matrix != Matrix.identity()
        assert isinstance(wp.updated, Signal)
        assert isinstance(wp.transform_changed, Signal)
        assert wp.tabs == []
        assert wp.tabs_enabled is True
        assert wp.source_segment is not None
        assert wp._edited_boundaries is None
        assert wp.geometry_provider_uid is None
        assert wp.geometry_provider_params == {}
        assert wp._transient_geometry_provider is None

        # New check for view cache initialization
        assert hasattr(wp, "_view_cache")
        assert isinstance(wp._view_cache, dict)

    def test_workpiece_is_docitem(self, workpiece_instance):
        assert isinstance(workpiece_instance, DocItem)
        assert hasattr(workpiece_instance, "get_world_transform")

    def test_geometry_provider_fields_serialization(self, workpiece_instance):
        """Test serialization of geometry_provider_uid and params fields."""
        wp = workpiece_instance

        wp.geometry_provider_params = {"width": 123.45, "height": 50.0}
        wp.geometry_provider_uid = "provider-123"

        data_dict = wp.to_dict()

        assert data_dict["geometry_provider_uid"] == "provider-123"
        assert data_dict["geometry_provider_params"] == {
            "width": 123.45,
            "height": 50.0,
        }

        new_wp = WorkPiece.from_dict(data_dict)

        assert new_wp.geometry_provider_uid == "provider-123"
        assert new_wp.geometry_provider_params == {
            "width": 123.45,
            "height": 50.0,
        }

    def test_boundaries_override(self, workpiece_instance):
        """
        Test that _edited_boundaries overrides the source segment geometry.
        """
        wp = workpiece_instance

        # Create a distinct edited geometry
        edited_geo = Geometry()
        edited_geo.move_to(0.2, 0.2)
        edited_geo.line_to(0.8, 0.8)

        # Initially, boundaries should come from source segment
        assert wp._edited_boundaries is None
        original_boundaries = wp.boundaries
        assert original_boundaries is not None
        assert original_boundaries is not edited_geo

        # Set override
        wp._edited_boundaries = edited_geo
        assert wp.boundaries is edited_geo

        # Clear override
        wp._edited_boundaries = None
        # Should revert to cached original (y-up version)
        assert wp.boundaries is not edited_geo

    def test_world_space_boundaries(self, workpiece_instance):
        """
        Test that world_space_boundaries returns geometry scaled to mm.
        """
        wp = workpiece_instance

        # Get normalized boundaries
        normalized = wp.boundaries
        assert normalized is not None
        norm_rect = normalized.rect()

        # Get world-space boundaries
        world = wp.world_space_boundaries
        assert world is not None
        world_rect = world.rect()

        # World rect should be normalized rect scaled by natural size
        w = wp.natural_width_mm
        h = wp.natural_height_mm

        assert world_rect[0] == pytest.approx(norm_rect[0] * w, rel=0.01)
        assert world_rect[1] == pytest.approx(norm_rect[1] * h, rel=0.01)
        assert world_rect[2] == pytest.approx(norm_rect[2] * w, rel=0.01)
        assert world_rect[3] == pytest.approx(norm_rect[3] * h, rel=0.01)

    def test_world_space_boundaries_empty_workpiece(self):
        """
        Test that world_space_boundaries returns None for empty workpiece.
        """
        wp = WorkPiece(name="empty")
        assert wp.boundaries is None
        assert wp.world_space_boundaries is None

    def test_apply_split(self, workpiece_instance):
        """Test the splitting logic."""
        wp = workpiece_instance
        # Set up a workpiece with two disjoint components in 0-1 space
        # Component 1: Rect at (0,0) size 0.2x0.2
        comp1 = Geometry()
        comp1.move_to(0, 0)
        comp1.line_to(0.2, 0)
        comp1.line_to(0.2, 0.2)
        comp1.line_to(0, 0.2)
        comp1.close_path()

        # Component 2: Rect at (0.5, 0.5) size 0.4x0.4
        comp2 = Geometry()
        comp2.move_to(0.5, 0.5)
        comp2.line_to(0.9, 0.5)
        comp2.line_to(0.9, 0.9)
        comp2.line_to(0.5, 0.9)
        comp2.close_path()

        # Set up WP dimensions: 100x100 at (0,0)
        wp.set_size(100, 100)
        wp.pos = (0, 0)

        fragments = [comp1, comp2]
        new_workpieces = wp.apply_split(fragments)

        assert len(new_workpieces) == 2
        wp1, wp2 = new_workpieces

        # Check WP1 (Derived from comp1)
        # Original size 100x100. Comp1 is 0.2x0.2 of that.
        # Expected physical size: 20x20
        # Expected pos: (0, 0)
        assert wp1.size == pytest.approx((20.0, 20.0))
        assert wp1.pos == pytest.approx((0.0, 0.0))

        # The split implementation creates clean segments using
        # edited_boundaries override
        assert wp1._edited_boundaries is not None
        assert wp1.boundaries is not None

        # Normalized boundary should fill 0-1 box
        min_x, min_y, max_x, max_y = wp1.boundaries.rect()
        assert min_x == pytest.approx(0.0)
        assert min_y == pytest.approx(0.0)
        assert max_x == pytest.approx(1.0)
        assert max_y == pytest.approx(1.0)

        # Check WP2 (Derived from comp2)
        # Original size 100x100. Comp2 is 0.4x0.4.
        # Expected physical size: 40x40
        # Expected pos: (50, 50)
        assert wp2.size == pytest.approx((40.0, 40.0))
        assert wp2.pos == pytest.approx((50.0, 50.0))

        # The split implementation creates clean segments using
        # edited_boundaries
        assert wp2._edited_boundaries is not None
        assert wp2.boundaries is not None

        min_x, min_y, max_x, max_y = wp2.boundaries.rect()
        assert min_x == pytest.approx(0.0)
        assert min_y == pytest.approx(0.0)
        assert max_x == pytest.approx(1.0)
        assert max_y == pytest.approx(1.0)

    def test_apply_split_filters_noise(self, workpiece_instance):
        """
        Tests that apply_split filters out fragments that are physically
        microscopic.
        """
        wp = workpiece_instance
        wp.set_size(100, 100)  # 100mm x 100mm

        # 1. Valid component (10mm x 10mm) -> 0.1 x 0.1 in normalized space
        valid = Geometry()
        valid.move_to(0, 0)
        valid.line_to(0.1, 0)
        valid.line_to(0.1, 0.1)
        valid.line_to(0, 0.1)
        valid.close_path()

        # 2. Noise component (0.0005mm x 0.0005mm) -> 0.000005 in normalized
        # space. Threshold is 0.1mm.
        noise = Geometry()
        noise.move_to(0.5, 0.5)
        noise.line_to(0.500005, 0.5)
        noise.line_to(0.500005, 0.500005)
        noise.line_to(0.5, 0.500005)
        noise.close_path()

        fragments = [valid, noise]
        new_workpieces = wp.apply_split(fragments)

        assert len(new_workpieces) == 1
        # Ensure the resulting WP corresponds to the valid fragment
        # (size ~ 10mm)
        assert new_workpieces[0].size == pytest.approx((10.0, 10.0))

    def test_serialization_with_tabs(self, workpiece_instance):
        """Tests that tabs are correctly serialized and deserialized."""
        wp = workpiece_instance
        wp.tabs = [
            Tab(width=3.0, segment_index=1, pos=0.5, uid="tab1"),
            Tab(width=3.0, segment_index=5, pos=0.25, uid="tab2"),
        ]
        wp.tabs_enabled = False

        data_dict = wp.to_dict()

        assert "tabs" in data_dict
        assert "tabs_enabled" in data_dict
        assert data_dict["tabs_enabled"] is False
        assert len(data_dict["tabs"]) == 2
        assert data_dict["tabs"][0] == asdict(wp.tabs[0])

        new_wp = WorkPiece.from_dict(data_dict)

        assert new_wp.tabs_enabled is False
        assert len(new_wp.tabs) == 2
        assert new_wp.tabs[0].uid == "tab1"
        assert new_wp.tabs[1].width == 3.0
        assert new_wp.tabs[1].pos == 0.25

    def test_setters_and_signals(self, workpiece_instance):
        wp = workpiece_instance
        updated_events, transform_events = [], []

        wp.updated.connect(
            lambda sender: updated_events.append(sender), weak=False
        )
        wp.transform_changed.connect(
            lambda sender, *, old_matrix=None: transform_events.append(sender),
            weak=False,
        )

        # pos setter should fire transform_changed, NOT updated.
        wp.pos = (10, 20)
        assert wp.pos == pytest.approx((10.0, 20.0))
        assert len(updated_events) == 0
        assert len(transform_events) == 1

        # set_size should fire transform_changed, NOT updated.
        wp.set_size(150, 75)
        assert wp.size == pytest.approx((150.0, 75.0))
        assert len(updated_events) == 0
        assert len(transform_events) == 2

        # angle setter should fire transform_changed, NOT updated.
        wp.angle = 45
        assert wp.angle == pytest.approx(45.0)
        assert len(updated_events) == 0
        assert len(transform_events) == 3

    def test_sizing_and_aspect_ratio(self, workpiece_instance):
        wp = workpiece_instance
        assert wp.get_natural_aspect_ratio() == pytest.approx(
            102.0 / 52.0, abs=1e-4
        )
        # get_default_size should return the SVG's natural size.
        assert wp.get_default_size(
            bounds_width=1000, bounds_height=1000
        ) == pytest.approx((102.0, 52.0), abs=1e-4)
        # The importer sets the size to the natural size.
        assert wp.size == pytest.approx((102.0, 52.0), abs=1e-4)

        wp.set_size(80, 20)
        assert wp.size == pytest.approx((80.0, 20.0))
        assert wp.get_current_aspect_ratio() == pytest.approx(4.0)

    def test_get_default_size_fallback(self):
        """
        Tests the fallback sizing logic when metadata is missing.
        """

        class MockNoSizeRenderer(SvgRenderer):
            def get_natural_size_from_data(self, **kwargs):
                return None

        # Setup doc and source with the mock renderer
        doc = Doc()
        source = SourceAsset(
            source_file=Path("nosize.dat"),
            original_data=b"",
            renderer=MockNoSizeRenderer(),
        )
        doc.add_asset(source)
        gen_config = SourceAssetSegment(
            source_asset_uid=source.uid,
            pristine_geometry=Geometry(),
            vectorization_spec=PassthroughSpec(),
        )
        wp = WorkPiece("nosize.dat", source_segment=gen_config)
        doc.active_layer.add_child(wp)

        # The size should fall back to the provided bounds
        assert wp.get_default_size(
            bounds_width=400.0, bounds_height=300.0
        ) == (400.0, 300.0)

    def test_render_for_ops(self, workpiece_instance):
        """Tests that render_for_ops renders at the current size."""
        wp = workpiece_instance
        wp.set_size(100, 50)
        surface = wp.render_for_ops(pixels_per_mm_x=10, pixels_per_mm_y=10)

        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 1000
        assert surface.get_height() == 500

    def test_render_chunk(self, workpiece_instance):
        """
        Tests that render_chunk yields chunks that correctly tile the full
        image area.
        """
        wp = workpiece_instance
        wp.set_size(100, 50)
        chunks = list(
            wp.render_chunk(
                pixels_per_mm_x=1,
                pixels_per_mm_y=1,
                max_chunk_width=40,
                max_chunk_height=40,
            )
        )
        assert len(chunks) == 6  # 3x2 grid of chunks
        max_x = max((c[1][0] + c[0].get_width() for c in chunks), default=0)
        assert max_x == 100
        max_y = max((c[1][1] + c[0].get_height() for c in chunks), default=0)
        assert max_y == 50

    def test_dump(self, workpiece_instance, capsys):
        """
        Tests the console output of the dump method.
        """
        workpiece_instance.dump(indent=1)
        captured = capsys.readouterr()

        # Check for components to avoid issues with temporary paths in output
        assert workpiece_instance.source_file.name in captured.out
        assert "SvgRenderer" in captured.out

    def test_get_world_transform_simple_translation(self, workpiece_instance):
        wp = workpiece_instance
        wp.pos = (10, 20)
        matrix = wp.get_world_transform()
        p_in = (0, 0)
        p_out = matrix.transform_point(p_in)
        assert p_out == pytest.approx((10, 20))

    def test_get_world_transform_scale(self, workpiece_instance):
        wp = workpiece_instance
        # set_size preserves center. Original center is ~ (50, 25).
        # New pos will be (50-10, 25-5) = (40, 20)
        wp.set_size(20, 10)
        matrix = wp.get_world_transform()
        p_in = (1, 1)  # Local corner in a 1x1 space
        p_out = matrix.transform_point(p_in)
        # After sizing, center is (50,25), pos is (40,20).
        # The new world transform is T(40,20) @ S(20,10).
        # It transforms local corner (1,1) to (40,20) + (20,10) = (60,30).
        assert p_out == pytest.approx((60.0, 30.0), abs=1)

    def test_get_world_transform_rotation(self, workpiece_instance):
        wp = workpiece_instance
        # set_size preserves center. Initial center (50, 25) moves to a pos
        # that keeps it at the same world coords.
        wp.set_size(20, 10)
        # angle.setter rebuilds the matrix, rotating around the scaled center.
        wp.angle = 90
        matrix = wp.get_world_transform()
        # Calculation is complex, but we can check a known point.
        # Center of 100x50 is (0.5,0.5) in local coords.
        # After sizing to 20x10, world center is (50,25).
        # Let's check the transformed center instead.
        center_out = matrix.transform_point((0.5, 0.5))
        assert center_out == pytest.approx((50, 25))

    def test_get_world_transform_all(self, workpiece_instance):
        wp = workpiece_instance
        wp.set_size(20, 10)
        wp.pos = (100, 200)
        wp.angle = 90
        matrix = wp.get_world_transform()

        p_in = (0, 0)
        p_out = matrix.transform_point(p_in)
        # Center of 20x10 is (10,5). Pos is (100,200). So center is (110,205).
        # Rotate origin (0,0) 90 deg around (10,5) -> (15, -5).
        # Add original pos -> (115, 195).
        assert p_out == pytest.approx((115, 195))

    def test_decomposed_properties_consistency(self, workpiece_instance):
        wp = workpiece_instance
        target_pos = (12.3, 45.6)
        target_size = (78.9, 101.1)
        target_angle = 33.3

        # Set properties once and check. The order matters.
        wp.set_size(*target_size)
        wp.angle = target_angle
        wp.pos = target_pos

        assert wp.pos == pytest.approx(target_pos, abs=1e-9)
        assert wp.size == pytest.approx(target_size, abs=1e-9)
        assert wp.angle == pytest.approx(target_angle, abs=1e-9)

        # Test again with a different order and values
        target_pos2 = (-5, -10)
        target_size2 = (20, 20)
        target_angle2 = 180

        # Set properties in a different order
        wp.angle = target_angle2
        wp.pos = target_pos2
        wp.set_size(*target_size2)

        # Check the final size and angle, which should be correct
        assert wp.size == pytest.approx(target_size2, abs=1e-9)
        assert wp.angle == pytest.approx(target_angle2, abs=1e-9)
        final_pos = wp.pos
        wp.pos = final_pos
        assert wp.pos == pytest.approx(final_pos, abs=1e-9)

    def test_negative_angle_preservation(self, workpiece_instance):
        """
        Tests that a negative angle is correctly set and retrieved.
        """
        wp = workpiece_instance
        wp.angle = -45.0
        assert wp.angle == pytest.approx(-45.0)

        wp.angle = -10.0
        assert wp.angle == pytest.approx(-10.0)

        # Also check a positive angle to ensure no regressions.
        wp.angle = 10.0
        assert wp.angle == pytest.approx(10.0)

    def test_get_tab_direction(self, workpiece_instance):
        wp = WorkPiece(name="test")
        # To get a CCW path in the final Y-up system, we define a Y-down path
        # that will be flipped by the .boundaries property. This is a CW path
        # in Y-down coords, which becomes CCW in Y-up.
        geo_y_down = Geometry()
        geo_y_down.move_to(0, 0)  # Top-left
        geo_y_down.line_to(1, 0)  # Top-right  (index 1)
        geo_y_down.line_to(1, 1)  # Btm-right (index 2)
        geo_y_down.line_to(0, 1)  # Btm-left  (index 3)
        geo_y_down.close_path()  # (index 4)
        segment = SourceAssetSegment(
            source_asset_uid="<none>",
            pristine_geometry=geo_y_down,
            normalization_matrix=Matrix.translation(0, 1)
            @ Matrix.scale(1, -1),
            vectorization_spec=PassthroughSpec(),
        )

        # Case 1: No source_segment means no boundaries
        wp.source_segment = None
        tab = Tab(width=1, segment_index=1, pos=0.5)
        assert wp.get_tab_direction(tab) is None
        wp.source_segment = segment  # Restore for subsequent tests

        # Case 2: No transform. Test the bottom edge of the final Y-up shape.
        # The Y-down path (0,1)->(1,1) (segment 3) becomes the Y-up path
        # (0,0)->(1,0), which is the bottom edge.
        wp.matrix = Matrix.identity()
        tab = Tab(width=1, segment_index=3, pos=0.5)
        direction = wp.get_tab_direction(tab)
        assert direction is not None
        assert direction == pytest.approx((0, -1))

        # Case 3: 90 degree rotation
        wp.angle = 90
        direction = wp.get_tab_direction(tab)
        assert direction is not None
        # Normal (0, -1) rotated by 90 deg becomes (1, 0)
        assert direction == pytest.approx((1, 0))

        # Case 4: Scale and rotation
        # The Y-down geo has a bbox of 1x1. set_size will scale it.
        wp.set_size(20, 10)
        wp.angle = -90
        direction = wp.get_tab_direction(tab)
        assert direction is not None
        # Normal (0, -1) rotated by -90 deg becomes (-1, 0)
        assert direction == pytest.approx((-1, 0))

        # Case 5: Open path
        open_geo_y_down = Geometry()
        open_geo_y_down.move_to(0, 0)
        open_geo_y_down.line_to(1, 0)
        wp.source_segment.pristine_geometry = open_geo_y_down
        wp.clear_render_cache()  # This will also clear the boundaries cache
        # Use a tab on the only existing segment (index 1)
        tab_on_open = Tab(width=1, segment_index=1, pos=0.5)
        assert wp.get_tab_direction(tab_on_open) is None

    def test_get_tab_direction_non_uniform_scale_diagonal(
        self, workpiece_instance
    ):
        """
        Tests that the tab normal is correct for a diagonal path under
        non-uniform scaling.
        """
        wp = WorkPiece(name="test")
        # We define a Y-down, CW path. The .boundaries property will flip
        # it to a Y-up, CCW path for processing. This is a 45-degree
        # rotated square.
        geo = Geometry()
        geo.move_to(1, 0)  # Top-center
        geo.line_to(2, 1)  # Segment 1: top-right side in Y-down
        geo.line_to(1, 2)  # bottom-center
        geo.line_to(0, 1)  # top-left
        geo.close_path()
        segment = SourceAssetSegment(
            source_asset_uid="<none>",
            pristine_geometry=geo,
            normalization_matrix=Matrix.translation(0, 1)
            @ Matrix.scale(1, -1),
            vectorization_spec=PassthroughSpec(),
        )
        wp.source_segment = segment

    def test_workpiece_forward_compatibility_with_extra_fields(self):
        """
        Tests that from_dict() preserves extra fields from newer versions
        and to_dict() re-serializes them.
        """
        wp_dict = {
            "uid": "wp-forward-456",
            "type": "workpiece",
            "name": "Future WorkPiece",
            "matrix": Matrix.identity().to_list(),
            "width_mm": 100.0,
            "height_mm": 50.0,
            "tabs": [],
            "tabs_enabled": True,
            "source_segment": None,
            "edited_boundaries": None,
            "sketch_uid": None,
            "sketch_params": {},
            "source_asset_uid": None,
            "future_field_string": "some value",
            "future_field_number": 42,
            "future_field_dict": {"nested": "data"},
        }

        wp = WorkPiece.from_dict(wp_dict)

        # Verify extra fields are stored
        assert wp.extra["future_field_string"] == "some value"
        assert wp.extra["future_field_number"] == 42
        assert wp.extra["future_field_dict"] == {"nested": "data"}

        # Verify extra fields are re-serialized
        data = wp.to_dict()
        assert data["future_field_string"] == "some value"
        assert data["future_field_number"] == 42
        assert data["future_field_dict"] == {"nested": "data"}

    def test_workpiece_backward_compat_missing_optional_fields(self):
        """
        Tests that from_dict() handles missing optional fields gracefully
        (simulating data from an older version).
        """
        minimal_dict = {
            "uid": "wp-backward-789",
            "type": "workpiece",
            "name": "Old WorkPiece",
            "matrix": Matrix.identity().to_list(),
            "width_mm": 100.0,
            "height_mm": 50.0,
            "tabs": [],
            "tabs_enabled": True,
            "source_segment": None,
            "edited_boundaries": None,
            "sketch_uid": None,
            "sketch_params": {},
            "source_asset_uid": None,
        }

        wp = WorkPiece.from_dict(minimal_dict)

        # Verify defaults are applied for missing optional fields
        assert wp.name == "Old WorkPiece"
        assert wp.extra == {}
