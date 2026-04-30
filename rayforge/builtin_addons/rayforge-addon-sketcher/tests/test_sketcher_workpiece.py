import pytest
from pathlib import Path
from rayforge.core.doc import Doc
from rayforge.core.geo import Geometry
from rayforge.core.matrix import Matrix
from rayforge.core.workpiece import WorkPiece
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from sketcher.core import Sketch


@pytest.fixture
def doc():
    """Provides a Doc instance."""
    return Doc()


@pytest.fixture
def doc_with_workpiece(doc):
    """Provides a doc with a workpiece."""
    from rayforge.image.svg.renderer import SVG_RENDERER
    from rayforge.core.source_asset import SourceAsset

    source = SourceAsset(
        source_file=Path("test.svg"),
        original_data=b"<svg></svg>",
        renderer=SVG_RENDERER,
    )
    doc.add_asset(source)

    segment = SourceAssetSegment(
        source_asset_uid=source.uid,
        pristine_geometry=Geometry(),
        normalization_matrix=Matrix.identity(),
        vectorization_spec=PassthroughSpec(),
    )

    wp = WorkPiece("test_wp")
    wp.source_segment = segment
    doc.active_layer.add_child(wp)

    return doc, wp, source


def make_sketch_with_geometry(width, height):
    """Create a sketch with a simple rectangle of given dimensions."""
    sketch = Sketch(name=f"Sketch {width}x{height}")
    p0 = sketch.origin_id
    p1 = sketch.add_point(width, 0)
    p2 = sketch.add_point(width, height)
    p3 = sketch.add_point(0, height)

    sketch.add_line(p0, p1)
    sketch.add_line(p1, p2)
    sketch.add_line(p2, p3)
    sketch.add_line(p3, p0)

    sketch.constrain_horizontal(p0, p1)
    sketch.constrain_vertical(p1, p2)
    sketch.constrain_horizontal(p3, p2)
    sketch.constrain_vertical(p0, p3)
    sketch.solve()

    return sketch


class TestWorkPieceWithSketch:
    """Tests for WorkPiece with Sketch geometry provider."""

    def test_serialization_deserialization(self, doc_with_workpiece):
        """
        Test serialization of geometry_provider_uid and params with Sketch.
        """
        doc, wp, source = doc_with_workpiece

        sketch = make_sketch_with_geometry(100, 50)
        doc.add_asset(sketch)
        wp.geometry_provider_uid = sketch.uid
        wp.geometry_provider_params = {"width": 123.45}

        wp.set_size(80.0, 40.0)
        wp.pos = (10.5, 20.2)
        wp.angle = 90

        data_dict = wp.to_dict()

        assert data_dict["geometry_provider_uid"] == sketch.uid
        assert data_dict["geometry_provider_params"] == {"width": 123.45}

        new_wp = WorkPiece.from_dict(data_dict)

        assert new_wp.geometry_provider_uid == sketch.uid
        assert new_wp.geometry_provider_params == {"width": 123.45}

    def test_in_world_hydrates_sketch_definition(self, doc_with_workpiece):
        """
        Tests that the in_world method correctly populates the transient
        sketch definition for use in subprocesses.
        """
        doc, wp, _ = doc_with_workpiece

        sketch = make_sketch_with_geometry(100, 50)
        doc.add_asset(sketch)
        wp.geometry_provider_uid = sketch.uid
        wp.geometry_provider_params = {"width": 50.0}

        world_wp = wp.in_world()

        assert world_wp.geometry_provider_uid == sketch.uid
        assert world_wp.geometry_provider_params == {"width": 50.0}

        transient_def = world_wp._transient_geometry_provider
        assert transient_def is not None
        assert isinstance(transient_def, Sketch)
        assert transient_def is not sketch
        assert transient_def.uid == sketch.uid

    def test_get_geometry_provider(self, doc_with_workpiece):
        """
        Tests retrieving the geometry provider from the document or from the
        transient field.
        """
        doc, wp, _ = doc_with_workpiece

        assert wp.geometry_provider_uid is None
        assert wp.get_geometry_provider() is None

        sketch_from_doc = make_sketch_with_geometry(100, 50)
        doc.add_asset(sketch_from_doc)
        wp.geometry_provider_uid = sketch_from_doc.uid

        retrieved_provider = wp.get_geometry_provider()
        assert retrieved_provider is not None
        assert retrieved_provider is sketch_from_doc
        assert retrieved_provider.uid == sketch_from_doc.uid

        transient_provider = make_sketch_with_geometry(50, 25)
        transient_provider.uid = "transient-uid-123"

        wp_with_transient = WorkPiece(
            "transient_test", source_segment=wp.source_segment
        )
        wp_with_transient.geometry_provider_uid = sketch_from_doc.uid
        wp_with_transient._transient_geometry_provider = transient_provider

        doc.active_layer.add_child(wp_with_transient)

        retrieved_transient = wp_with_transient.get_geometry_provider()
        assert retrieved_transient is not None
        assert retrieved_transient is transient_provider
        assert retrieved_transient is not sketch_from_doc
        assert retrieved_transient.uid == "transient-uid-123"

    def test_from_geometry_provider_factory_behavior(self):
        """
        Tests the WorkPiece.from_geometry_provider factory method logic.
        """
        sketch = make_sketch_with_geometry(10, 20)

        wp = WorkPiece.from_geometry_provider(sketch)

        assert wp.geometry_provider_uid == sketch.uid
        assert wp.name == sketch.name
        assert wp.natural_width_mm == pytest.approx(10.0)
        assert wp.natural_height_mm == pytest.approx(20.0)

        sx, sy = wp.matrix.get_scale()
        assert sx == pytest.approx(10.0)
        assert sy == pytest.approx(20.0)

        empty_sketch = Sketch(name="Empty")
        empty_sketch.solve()
        wp_empty = WorkPiece.from_geometry_provider(empty_sketch)

        assert wp_empty.natural_width_mm == 0.0
        assert wp_empty.natural_height_mm == 0.0

    def test_sketch_params_setter_triggers_update(self, doc_with_workpiece):
        """
        Tests that setting geometry_provider_params triggers regeneration
        and updates natural dimensions.
        """
        doc, wp, _ = doc_with_workpiece

        sketch = Sketch(name="ParametricSketch")
        sketch.set_param("W", 100)
        sketch.set_param("H", 50)

        p0 = sketch.origin_id
        p1 = sketch.add_point(100, 0)
        p2 = sketch.add_point(100, 50)
        p3 = sketch.add_point(0, 50)

        sketch.add_line(p0, p1)
        sketch.add_line(p1, p2)
        sketch.add_line(p2, p3)
        sketch.add_line(p3, p0)

        sketch.constrain_horizontal(p0, p1)
        sketch.constrain_vertical(p1, p2)
        sketch.constrain_horizontal(p3, p2)
        sketch.constrain_vertical(p0, p3)
        sketch.constrain_distance(p0, p1, "W")
        sketch.constrain_distance(p0, p3, "H")
        sketch.solve()

        doc.add_asset(sketch)
        wp.geometry_provider_uid = sketch.uid

        updated_signals = []
        wp.updated.connect(lambda s: updated_signals.append(s), weak=False)

        new_params = {"W": 50, "H": 25}
        wp.geometry_provider_params = new_params

        assert wp.geometry_provider_params == new_params
        assert len(updated_signals) > 0

        assert wp.natural_width_mm == pytest.approx(50.0)
        assert wp.natural_height_mm == pytest.approx(25.0)

    def test_sketch_boundaries_normalization(self, doc_with_workpiece):
        """
        Tests that boundaries generated from a sketch are correctly normalized
        to the 0-1 unit square, regardless of the sketch's physical size.
        """
        doc, wp, _ = doc_with_workpiece

        sketch = make_sketch_with_geometry(100, 200)

        doc.add_asset(sketch)
        wp.geometry_provider_uid = sketch.uid
        wp.clear_render_cache()

        bounds = wp.boundaries
        assert bounds is not None

        min_x, min_y, max_x, max_y = bounds.rect()
        assert min_x == pytest.approx(0.0)
        assert min_y == pytest.approx(0.0)
        assert max_x == pytest.approx(1.0)
        assert max_y == pytest.approx(1.0)

        assert wp._boundaries_cache is bounds

    def test_uuid4_unique_per_workpiece_instance(self, doc):
        """
        Two workpiece instances from the same sketch each get their own
        uuid4 value, and it stays stable across boundary accesses.
        """
        from rayforge.core.geo.font_config import FontConfig

        sketch = Sketch(name="UUID Sketch")
        origin = sketch.add_point(0, 0)
        w_pt = sketch.add_point(10, 0)
        h_pt = sketch.add_point(0, 10)
        sketch.registry.add_text_box(
            origin, w_pt, h_pt, "{uuid4()}", FontConfig()
        )
        sketch.solve()
        doc.add_asset(sketch)

        wp_a = WorkPiece.from_geometry_provider(sketch)
        doc.active_layer.add_child(wp_a)

        wp_b = WorkPiece.from_geometry_provider(sketch)
        doc.active_layer.add_child(wp_b)

        assert wp_a._resolved_text_cache != {}
        assert wp_b._resolved_text_cache != {}
        cache_a_val = next(iter(wp_a._resolved_text_cache.values()))
        cache_b_val = next(iter(wp_b._resolved_text_cache.values()))
        assert cache_a_val != cache_b_val

    def test_uuid4_stable_within_workpiece(self, doc):
        """
        A workpiece instance returns the same uuid4 on repeated
        boundary accesses (cache survives).
        """
        from rayforge.core.geo.font_config import FontConfig

        sketch = Sketch(name="UUID Sketch")
        origin = sketch.add_point(0, 0)
        w_pt = sketch.add_point(10, 0)
        h_pt = sketch.add_point(0, 10)
        sketch.registry.add_text_box(
            origin, w_pt, h_pt, "{uuid4()}", FontConfig()
        )
        sketch.solve()
        doc.add_asset(sketch)

        wp = WorkPiece.from_geometry_provider(sketch)
        doc.active_layer.add_child(wp)

        first_cache = dict(wp._resolved_text_cache)
        wp.clear_render_cache()
        wp._boundaries_cache = None
        wp._fills_cache = None
        _ = wp.boundaries
        assert wp._resolved_text_cache == first_cache

    def test_uuid4_survives_in_world(self, doc):
        """
        in_world carries the resolved_text_cache so the subprocess
        uses the same uuid4.
        """
        from rayforge.core.geo.font_config import FontConfig

        sketch = Sketch(name="UUID Sketch")
        origin = sketch.add_point(0, 0)
        w_pt = sketch.add_point(10, 0)
        h_pt = sketch.add_point(0, 10)
        sketch.registry.add_text_box(
            origin, w_pt, h_pt, "{uuid4()}", FontConfig()
        )
        sketch.solve()
        doc.add_asset(sketch)

        wp = WorkPiece.from_geometry_provider(sketch)
        doc.active_layer.add_child(wp)
        original_cache = dict(wp._resolved_text_cache)

        world = wp.in_world()
        assert world._resolved_text_cache == original_cache

    def test_uuid4_survives_serialization(self, doc):
        """
        The resolved_text_cache round-trips through
        to_dict / from_dict.
        """
        from rayforge.core.geo.font_config import FontConfig

        sketch = Sketch(name="UUID Sketch")
        origin = sketch.add_point(0, 0)
        w_pt = sketch.add_point(10, 0)
        h_pt = sketch.add_point(0, 10)
        sketch.registry.add_text_box(
            origin, w_pt, h_pt, "{uuid4()}", FontConfig()
        )
        sketch.solve()
        doc.add_asset(sketch)

        wp = WorkPiece.from_geometry_provider(sketch)
        doc.active_layer.add_child(wp)
        original_cache = dict(wp._resolved_text_cache)

        data = wp.to_dict()
        wp2 = WorkPiece.from_dict(data)
        assert wp2._resolved_text_cache == original_cache

    def test_uuid4_cleared_on_sketch_edit(self, doc):
        """
        When the sketch is edited, the workpiece's cache is cleared
        and a new uuid4 is generated.
        """
        from rayforge.core.geo.font_config import FontConfig

        sketch = Sketch(name="UUID Sketch")
        origin = sketch.add_point(0, 0)
        w_pt = sketch.add_point(10, 0)
        h_pt = sketch.add_point(0, 10)
        sketch.registry.add_text_box(
            origin, w_pt, h_pt, "{uuid4()}", FontConfig()
        )
        sketch.solve()
        doc.add_asset(sketch)

        wp = WorkPiece.from_geometry_provider(sketch)
        doc.active_layer.add_child(wp)
        first_val = next(iter(wp._resolved_text_cache.values()))

        wp.get_geometry_provider()

        sketch.solve()
        sketch.notify_update()

        _ = wp.boundaries
        new_val = next(iter(wp._resolved_text_cache.values()))
        assert new_val != first_val
