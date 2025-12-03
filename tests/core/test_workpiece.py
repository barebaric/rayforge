import cairo
import pytest
from pathlib import Path
from typing import Tuple, cast
from dataclasses import asdict
from unittest.mock import MagicMock, patch
from blinker import Signal
from rayforge.core.doc import Doc
from rayforge.core.source_asset import SourceAsset
from rayforge.core.item import DocItem
from rayforge.core.matrix import Matrix
from rayforge.core.tab import Tab
from rayforge.core.geo import Geometry
from rayforge.core.workpiece import WorkPiece
from rayforge.core.sketcher.sketch import Sketch
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
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
    payload = import_file(svg_file)

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
        # This test is no longer valid, as a simple SVG might not be modified
        # by the importer, so base_render_data can equal original_data.
        # assert str(wp.data) != str(wp.source.original_data)
        assert sample_svg_data in wp.source.original_data
        assert wp.pos == pytest.approx((0.0, 0.0))
        assert wp.size == pytest.approx((100.0, 50.0))
        assert wp.angle == pytest.approx(0.0)
        # Importer sets size, so matrix is not identity
        assert wp.matrix != Matrix.identity()
        assert isinstance(wp.updated, Signal)
        assert isinstance(wp.transform_changed, Signal)
        assert wp.tabs == []
        assert wp.tabs_enabled is True
        assert wp.source_segment is not None
        assert wp._edited_boundaries is None
        assert wp.sketch_uid is None
        assert wp.sketch_params == {}
        assert wp._transient_sketch_definition is None

        # New check for view cache initialization
        assert hasattr(wp, "_view_cache")
        assert isinstance(wp._view_cache, dict)

    def test_workpiece_is_docitem(self, workpiece_instance):
        assert isinstance(workpiece_instance, DocItem)
        assert hasattr(workpiece_instance, "get_world_transform")

    def test_serialization_deserialization(self, doc_with_workpiece):
        doc, wp, source = doc_with_workpiece

        # Setup sketch data for serialization
        sketch = Sketch()
        doc.add_asset(sketch)
        wp.sketch_uid = sketch.uid
        wp.sketch_params = {"width": 123.45}

        wp.set_size(80.0, 40.0)
        wp.pos = (10.5, 20.2)
        wp.angle = 90

        # Set edited boundaries to verify persistence
        edited_geo = Geometry()
        edited_geo.move_to(0.1, 0.1)
        edited_geo.line_to(0.9, 0.9)
        wp._edited_boundaries = edited_geo

        data_dict = wp.to_dict()

        assert "renderer_name" not in data_dict
        assert "source_file" not in data_dict
        assert "data" not in data_dict
        assert "size" not in data_dict
        assert "boundaries" not in data_dict
        assert isinstance(data_dict["matrix"], list)
        assert "source_segment" in data_dict
        assert data_dict["source_segment"]["source_asset_uid"] == source.uid
        assert "edited_boundaries" in data_dict
        assert data_dict["sketch_uid"] == sketch.uid
        assert data_dict["sketch_params"] == {"width": 123.45}

        new_wp = WorkPiece.from_dict(data_dict)

        # A free-floating workpiece cannot access its source properties
        assert new_wp.data is None
        assert new_wp.source is None
        assert new_wp.source_file is None

        # Add it to the doc to link it to its source
        doc.active_layer.add_child(new_wp)

        assert new_wp.name == wp.name
        assert new_wp.source_file == source.source_file
        assert new_wp.source is not None
        assert isinstance(new_wp.source.renderer, SvgRenderer)
        assert new_wp.pos == pytest.approx(wp.pos)
        assert new_wp.size == pytest.approx(wp.size)
        assert new_wp.angle == pytest.approx(wp.angle, abs=1e-9)
        assert new_wp.matrix == wp.matrix
        assert new_wp.natural_size == (100.0, 50.0)
        assert new_wp.source_segment is not None
        assert new_wp.source_segment.source_asset_uid == source.uid
        assert new_wp._edited_boundaries is not None
        assert len(new_wp._edited_boundaries.commands) == len(
            edited_geo.commands
        )
        assert new_wp.sketch_uid == sketch.uid
        assert new_wp.sketch_params == {"width": 123.45}

    def test_in_world_hydrates_sketch_definition(self, doc_with_workpiece):
        """
        Tests that the in_world method correctly populates the transient
        sketch definition for use in subprocesses.
        """
        doc, wp, _ = doc_with_workpiece

        # Create and link a sketch
        sketch = Sketch()
        # Create a valid dictionary from a default sketch to use for mocking.
        mock_dict = Sketch().to_dict()
        mock_dict["uid"] = sketch.uid  # Ensure the mock uses the correct UID

        with patch.object(Sketch, "to_dict", return_value=mock_dict):
            doc.add_asset(sketch)
            wp.sketch_uid = sketch.uid
            wp.sketch_params = {"width": 50.0}

            # The method under test
            world_wp = wp.in_world()

        # Check that the main properties are copied
        assert world_wp.sketch_uid == sketch.uid
        assert world_wp.sketch_params == {"width": 50.0}

        # Check that the transient definition is hydrated correctly
        transient_def = world_wp._transient_sketch_definition
        assert transient_def is not None
        assert isinstance(transient_def, Sketch)
        assert transient_def is not sketch  # Must be a copy
        assert transient_def.uid == sketch.uid

    def test_get_sketch_definition(self, doc_with_workpiece):
        """
        Tests retrieving the sketch definition from the document or from the
        transient field.
        """
        doc, wp, _ = doc_with_workpiece

        # Case 1: No sketch UID is set, should return None
        assert wp.sketch_uid is None
        assert wp.get_sketch_definition() is None

        # Case 2: Sketch UID is set, should retrieve the sketch from the
        # document
        sketch_from_doc = Sketch()
        doc.add_asset(sketch_from_doc)
        wp.sketch_uid = sketch_from_doc.uid

        retrieved_sketch = wp.get_sketch_definition()
        assert retrieved_sketch is not None
        # It should be the exact same instance from the document's registry
        assert retrieved_sketch is sketch_from_doc
        assert retrieved_sketch.uid == sketch_from_doc.uid

        # Case 3: A transient sketch definition exists and takes precedence
        # This simulates a WorkPiece that has been prepared by `in_world()`
        # for use in a subprocess.
        transient_sketch = Sketch()
        transient_sketch.uid = "transient-uid-123"

        # Use a new WorkPiece instance to avoid side effects
        wp_with_transient = WorkPiece(
            "transient_test", source_segment=wp.source_segment
        )
        # Assign the doc-based sketch_uid to prove the transient one is used
        # instead
        wp_with_transient.sketch_uid = sketch_from_doc.uid
        wp_with_transient._transient_sketch_definition = transient_sketch

        # Add it to the doc so that a doc lookup *could* work, but shouldn't.
        doc.active_layer.add_child(wp_with_transient)

        retrieved_transient = wp_with_transient.get_sketch_definition()
        assert retrieved_transient is not None
        # It should be the transient instance, not the one from the doc
        assert retrieved_transient is transient_sketch
        assert retrieved_transient is not sketch_from_doc
        assert retrieved_transient.uid == "transient-uid-123"

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

        # The split implementation creates clean segments instead of using
        # edited_boundaries
        assert wp1._edited_boundaries is None
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

        assert wp2._edited_boundaries is None
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
            lambda sender: transform_events.append(sender), weak=False
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
        assert wp.get_natural_aspect_ratio() == pytest.approx(2.0)
        # get_default_size should return the SVG's natural size.
        assert wp.get_default_size(bounds_width=1000, bounds_height=1000) == (
            100.0,
            50.0,
        )
        # The importer sets the size to the natural size.
        assert wp.size == pytest.approx((100.0, 50.0))

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
            segment_mask_geometry=Geometry(),
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
        max_y = max((c[1][1] + c[0].get_height() for c in chunks), default=0)
        assert max_x == 100
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
        # set_size preserves center. Original center is (50, 25).
        # New pos will be (50-10, 25-5) = (40, 20)
        wp.set_size(20, 10)
        matrix = wp.get_world_transform()
        p_in = (1, 1)  # Local corner in a 1x1 space
        p_out = matrix.transform_point(p_in)
        # After sizing, center is (50,25), pos is (40,20).
        # The new world transform is T(40,20) @ S(20,10).
        # It transforms local corner (1,1) to (40,20) + (20,10) = (60,30).
        assert p_out == pytest.approx((60.0, 30.0))

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
            segment_mask_geometry=geo_y_down,
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
        wp.source_segment.segment_mask_geometry = open_geo_y_down
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
            segment_mask_geometry=geo,
            vectorization_spec=PassthroughSpec(),
        )
        wp.source_segment = segment

    def test_from_sketch_factory_behavior(self):
        """
        Tests the WorkPiece.from_sketch factory method logic.
        """
        # We patch the Sketch class inside rayforge.core.sketcher.sketch
        # because the module under test (workpiece.py) imports it locally.
        with patch("rayforge.core.sketcher.sketch.Sketch") as MockSketchCls:
            # Setup the mock instance returned by Sketch.from_dict()
            mock_instance = MagicMock()
            MockSketchCls.from_dict.return_value = mock_instance

            # Setup the geometry returned by the mock instance.
            # Define a 10x20 bounding box: (0,0) -> (10,20)
            mock_geo = Geometry()
            mock_geo.move_to(0, 0)
            mock_geo.line_to(10, 20)

            mock_instance.to_geometry.return_value = mock_geo
            mock_instance.solve.return_value = True
            mock_instance.name = "TestSketch"

            # Create a dummy sketch object to pass in (the factory calls
            # to_dict on it)
            dummy_sketch = Sketch()
            dummy_sketch.uid = "sketch-123"
            dummy_sketch.name = "TestSketch"

            # Run factory
            wp = WorkPiece.from_sketch(dummy_sketch)

            # Assertions
            assert wp.sketch_uid == dummy_sketch.uid
            assert wp.name == "TestSketch"
            assert wp.natural_width_mm == pytest.approx(10.0)
            assert wp.natural_height_mm == pytest.approx(20.0)

            # The matrix should be scaled to natural size
            sx, sy = wp.matrix.get_scale()
            assert sx == pytest.approx(10.0)
            assert sy == pytest.approx(20.0)

            # 2. Test empty/failing sketch fallback
            mock_instance.to_geometry.return_value = Geometry()  # Empty
            wp_empty = WorkPiece.from_sketch(dummy_sketch)

            assert wp_empty.natural_width_mm == 1.0
            assert wp_empty.natural_height_mm == 1.0

    def test_sketch_params_setter_triggers_update(self, doc_with_workpiece):
        """
        Tests that setting sketch_params triggers regeneration and updates
        natural dimensions.
        """
        doc, wp, _ = doc_with_workpiece

        # We create a mock sketch that returns specific geometry when solved
        sketch = Sketch()
        doc.add_asset(sketch)
        wp.sketch_uid = sketch.uid

        updated_signals = []
        wp.updated.connect(lambda s: updated_signals.append(s), weak=False)

        # Patch Sketch at its definition because it is imported locally
        # inside regenerate_from_sketch.
        with patch("rayforge.core.sketcher.sketch.Sketch") as MockSketchCls:
            # Setup the mock instance returned by Sketch.from_dict()
            mock_instance = MagicMock()
            MockSketchCls.from_dict.return_value = mock_instance

            # Setup the geometry returned by the mock instance
            mock_geo = Geometry()
            # Define a bounding box (0,0) -> (50,25)
            mock_geo.move_to(0, 0)
            mock_geo.line_to(50, 25)

            mock_instance.to_geometry.return_value = mock_geo
            mock_instance.solve.return_value = True

            # Action: Change params
            new_params = {"W": 50, "H": 25}
            wp.sketch_params = new_params

            # Assertions
            assert wp.sketch_params == new_params
            assert len(updated_signals) > 0

            # Verify regenerate logic was called
            mock_instance.solve.assert_called()
            # Verify variable overrides were passed to solve
            call_args = mock_instance.solve.call_args
            assert call_args.kwargs.get("variable_overrides") == new_params

            # Verify natural size updated
            assert wp.natural_width_mm == pytest.approx(50.0)
            assert wp.natural_height_mm == pytest.approx(25.0)

    def test_sketch_boundaries_normalization(self, doc_with_workpiece):
        """
        Tests that boundaries generated from a sketch are correctly normalized
        to the 0-1 unit square, regardless of the sketch's physical size.
        """
        doc, wp, _ = doc_with_workpiece

        sketch = Sketch()
        doc.add_asset(sketch)
        wp.sketch_uid = sketch.uid
        wp.clear_render_cache()

        # Patch Sketch at its definition
        with patch("rayforge.core.sketcher.sketch.Sketch") as MockSketchCls:
            # Setup mock instance
            mock_instance = MagicMock()
            MockSketchCls.from_dict.return_value = mock_instance

            # Setup the geometry: a 100x200 rectangle starting at 0,0
            # This is what wp.boundaries will process.
            mock_geo = Geometry()
            mock_geo.move_to(0, 0)
            mock_geo.line_to(100, 200)

            mock_instance.to_geometry.return_value = mock_geo
            mock_instance.solve.return_value = True

            # Access boundaries
            bounds = wp.boundaries
            assert bounds is not None

            # Verify normalization (should fill 0-1 box)
            # The logic inside WorkPiece.boundaries detects the 100x200
            # extent and normalizes it.
            min_x, min_y, max_x, max_y = bounds.rect()
            assert min_x == pytest.approx(0.0)
            assert min_y == pytest.approx(0.0)
            assert max_x == pytest.approx(1.0)
            assert max_y == pytest.approx(1.0)

            # Check cache was populated
            assert wp._boundaries_cache is bounds
