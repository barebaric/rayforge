"""
Tests that layout/flip commands honour the PANEL presentation rotation.

The document model lives in WORLD space while the 2D canvas presents
PANEL space. Operations that act on what the user sees (align, spread,
flip, center) must therefore compute their targets in PANEL coordinates
and un-rotate the resulting deltas back into WORLD space.
"""

import pytest
from raygeo.geo import Matrix

from rayforge.core.layer import Layer
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.layout.align import (
    BboxAlignBottomStrategy,
    BboxAlignCenterStrategy,
    BboxAlignLeftStrategy,
    BboxAlignMiddleStrategy,
    BboxAlignRightStrategy,
    BboxAlignTopStrategy,
)
from rayforge.doceditor.layout.spread import (
    SpreadHorizontallyStrategy,
    SpreadVerticallyStrategy,
)
from rayforge.doceditor.transform_cmd import TransformCmd
from rayforge.machine.models.machine_panel import PanelOrientation


@pytest.fixture
def active_machine(lite_context, sync_machine):
    """A real machine on a 400x300 native bed, active in the global
    context so the layout strategies pick it up via get_context()."""
    sync_machine.set_axis_extents(400, 300)
    lite_context.config.set_machine(sync_machine)
    return sync_machine


def _workpiece(layer: Layer, name: str, pos, size) -> WorkPiece:
    wp = WorkPiece(name=name)
    wp.set_size(*size)
    wp.pos = pos
    layer.add_child(wp)
    return wp


def _apply(strategy, items):
    """Calculate the strategy deltas against the active machine and apply
    them to the items."""
    deltas = strategy.calculate_deltas()
    for item, delta in deltas.items():
        item.matrix = delta @ item.matrix
    return deltas


def _panel_bbox(machine, item):
    """Panel-space bbox of a single item's unit square."""
    world_transform = item.get_world_transform()
    corners = [
        world_transform.transform_point(point)
        for point in ((0, 0), (1, 0), (1, 1), (0, 1))
    ]
    matrix = machine.panel.get_world_to_panel_2d()
    xs, ys = [], []
    for px, py in (matrix.transform_point(point) for point in corners):
        xs.append(px)
        ys.append(py)
    return (min(xs), min(ys), max(xs), max(ys))


# A 20x10 item at world (100, 50) on a ROTATED_RIGHT bed occupies panel
# bbox (50, 280, 60, 300); at world (200, 100) it occupies
# (100, 180, 110, 200). World->panel for ROTATED_RIGHT is (X, Y) ->
# (Y, 400 - X); panel deltas un-rotate as (dx, dy) -> (-dy, dx).


class TestPanelAwareAlign:
    def test_align_left_rotated_right(self, active_machine):
        active_machine.panel.set_orientation(PanelOrientation.ROTATED_RIGHT)
        layer = Layer(name="L")
        a = _workpiece(layer, "a", (100, 50), (20, 10))

        _apply(BboxAlignLeftStrategy([a]), [a])

        assert _panel_bbox(active_machine, a) == pytest.approx(
            (0.0, 280.0, 10.0, 300.0)
        )

    def test_align_right_rotated_right(self, active_machine):
        active_machine.panel.set_orientation(PanelOrientation.ROTATED_RIGHT)
        layer = Layer(name="L")
        a = _workpiece(layer, "a", (200, 100), (20, 10))

        _apply(BboxAlignRightStrategy([a], surface_width_mm=300.0), [a])

        assert _panel_bbox(active_machine, a) == pytest.approx(
            (290.0, 180.0, 300.0, 200.0)
        )

    def test_align_top_rotated_right(self, active_machine):
        active_machine.panel.set_orientation(PanelOrientation.ROTATED_RIGHT)
        layer = Layer(name="L")
        a = _workpiece(layer, "a", (200, 100), (20, 10))

        _apply(BboxAlignTopStrategy([a], surface_height_mm=400.0), [a])

        assert _panel_bbox(active_machine, a) == pytest.approx(
            (100.0, 380.0, 110.0, 400.0)
        )

    def test_align_bottom_rotated_right(self, active_machine):
        active_machine.panel.set_orientation(PanelOrientation.ROTATED_RIGHT)
        layer = Layer(name="L")
        a = _workpiece(layer, "a", (200, 100), (20, 10))

        _apply(BboxAlignBottomStrategy([a]), [a])

        assert _panel_bbox(active_machine, a) == pytest.approx(
            (100.0, 0.0, 110.0, 20.0)
        )

    def test_center_horizontally_rotated_right(self, active_machine):
        active_machine.panel.set_orientation(PanelOrientation.ROTATED_RIGHT)
        layer = Layer(name="L")
        a = _workpiece(layer, "a", (200, 100), (20, 10))

        _apply(BboxAlignCenterStrategy([a], surface_width_mm=300.0), [a])

        assert _panel_bbox(active_machine, a) == pytest.approx(
            (145.0, 180.0, 155.0, 200.0)
        )

    def test_center_vertically_rotated_right(self, active_machine):
        active_machine.panel.set_orientation(PanelOrientation.ROTATED_RIGHT)
        layer = Layer(name="L")
        a = _workpiece(layer, "a", (200, 100), (20, 10))

        _apply(BboxAlignMiddleStrategy([a], surface_height_mm=400.0), [a])

        assert _panel_bbox(active_machine, a) == pytest.approx(
            (100.0, 190.0, 110.0, 210.0)
        )

    def test_align_left_multi_rotated_right(self, active_machine):
        active_machine.panel.set_orientation(PanelOrientation.ROTATED_RIGHT)
        layer = Layer(name="L")
        a = _workpiece(layer, "a", (100, 50), (20, 10))
        b = _workpiece(layer, "b", (300, 150), (20, 10))

        _apply(BboxAlignLeftStrategy([a, b]), [a, b])

        assert _panel_bbox(active_machine, a) == pytest.approx(
            (50.0, 280.0, 60.0, 300.0)
        )
        assert _panel_bbox(active_machine, b) == pytest.approx(
            (50.0, 80.0, 60.0, 100.0)
        )

    def test_align_native_matches_world(self, active_machine):
        layer = Layer(name="L")
        a = _workpiece(layer, "a", (100, 50), (20, 10))

        _apply(BboxAlignLeftStrategy([a]), [a])

        assert _panel_bbox(active_machine, a) == pytest.approx(
            (0.0, 50.0, 20.0, 60.0)
        )


class TestPanelAwareSpread:
    def test_spread_horizontally_rotated_right(self, active_machine):
        active_machine.panel.set_orientation(PanelOrientation.ROTATED_RIGHT)
        layer = Layer(name="L")
        # Placed along world Y so they line up across the panel X axis.
        a = _workpiece(layer, "a", (200, 100), (20, 10))
        b = _workpiece(layer, "b", (200, 200), (20, 10))
        c = _workpiece(layer, "c", (200, 50), (20, 10))

        _apply(SpreadHorizontallyStrategy([a, b, c]), [a, b, c])

        # Panel left edges are evenly gapped: 50, 125, 200.
        lefts = sorted(
            _panel_bbox(active_machine, item)[0] for item in (a, b, c)
        )
        assert lefts == pytest.approx([50.0, 125.0, 200.0])

    def test_spread_vertically_rotated_right(self, active_machine):
        active_machine.panel.set_orientation(PanelOrientation.ROTATED_RIGHT)
        layer = Layer(name="L")
        # Placed along world X so they line up across the panel Y axis.
        a = _workpiece(layer, "a", (100, 200), (20, 10))
        b = _workpiece(layer, "b", (200, 200), (20, 10))
        c = _workpiece(layer, "c", (50, 200), (20, 10))

        _apply(SpreadVerticallyStrategy([a, b, c]), [a, b, c])

        # Panel bottom edges are evenly gapped.
        bottoms = sorted(
            _panel_bbox(active_machine, item)[1] for item in (a, b, c)
        )
        assert bottoms == pytest.approx([180.0, 255.0, 330.0])


class TestPanelAwareFlip:
    def test_flip_horizontal_rotated_right(self, active_machine):
        """Flip mirrors along the presented vertical axis through the
        item's centre, keeping the on-screen bbox and centre fixed."""
        active_machine.panel.set_orientation(PanelOrientation.ROTATED_RIGHT)
        layer = Layer(name="L")
        a = _workpiece(layer, "a", (100, 50), (20, 10))

        bbox_before = _panel_bbox(active_machine, a)
        corner_before = (
            active_machine.panel.get_world_to_panel_2d().transform_point(
                a.get_world_transform().transform_point((0.0, 0.0))
            )
        )

        a.matrix = (
            TransformCmd._flip_matrix_world(a, horizontal=True) @ a.matrix
        )

        corner_after = (
            active_machine.panel.get_world_to_panel_2d().transform_point(
                a.get_world_transform().transform_point((0.0, 0.0))
            )
        )
        center_x = (bbox_before[0] + bbox_before[2]) / 2
        assert corner_after == pytest.approx(
            (2 * center_x - corner_before[0], corner_before[1])
        )
        # The on-screen bounding box is unchanged.
        assert _panel_bbox(active_machine, a) == pytest.approx(bbox_before)

    def test_flip_vertical_rotated_right(self, active_machine):
        active_machine.panel.set_orientation(PanelOrientation.ROTATED_RIGHT)
        layer = Layer(name="L")
        a = _workpiece(layer, "a", (100, 50), (20, 10))

        bbox_before = _panel_bbox(active_machine, a)
        corner_before = (
            active_machine.panel.get_world_to_panel_2d().transform_point(
                a.get_world_transform().transform_point((0.0, 0.0))
            )
        )

        a.matrix = (
            TransformCmd._flip_matrix_world(a, horizontal=False) @ a.matrix
        )

        corner_after = (
            active_machine.panel.get_world_to_panel_2d().transform_point(
                a.get_world_transform().transform_point((0.0, 0.0))
            )
        )
        center_y = (bbox_before[1] + bbox_before[3]) / 2
        assert corner_after == pytest.approx(
            (corner_before[0], 2 * center_y - corner_before[1])
        )
        assert _panel_bbox(active_machine, a) == pytest.approx(bbox_before)

    def test_flip_native_matches_world(self, active_machine):
        layer = Layer(name="L")
        a = _workpiece(layer, "a", (100, 50), (20, 10))

        corner_before = a.get_world_transform().transform_point((0.0, 0.0))
        a.matrix = (
            TransformCmd._flip_matrix_world(a, horizontal=True) @ a.matrix
        )
        corner_after = a.get_world_transform().transform_point((0.0, 0.0))

        # Native flip mirrors about the world centre x=110.
        assert corner_after == pytest.approx(
            (220.0 - corner_before[0], corner_before[1])
        )


def test_flip_matrix_is_reflection(active_machine):
    """The composed panel flip stays a pure reflection (det -1)."""
    active_machine.panel.set_orientation(PanelOrientation.ROTATED_RIGHT)

    world_to_panel = active_machine.panel.get_world_to_panel_2d()
    panel_to_world = world_to_panel.invert()
    center = world_to_panel.transform_point((110.0, 55.0))
    flip = (
        panel_to_world @ Matrix.flip_horizontal(center=center) @ world_to_panel
    )

    # A reflection applied twice is the identity transform.
    composed = flip @ flip
    for point in ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)):
        assert composed.transform_point(point) == pytest.approx(point)
