"""
Integration tests for ArrayCmd.

These build a real DocEditor with multiple layers and selections that
span them, then verify the array creates the right copies on the right
layers, with correct positions, and that everything is undoable.
"""

import pytest

from rayforge.core.layer import Layer
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.array import (
    ArrayMode,
    ArrayParams,
    CircularArrayParams,
    GridArrayParams,
    PointRotationParams,
    SpacingMode,
)
from rayforge.doceditor.array_cmd import ArrayCmd


@pytest.fixture
def array_cmd(doc_editor):
    """Provides an ArrayCmd linked to the doc_editor."""
    return ArrayCmd(doc_editor)


@pytest.fixture
def multilayer_selection(doc_editor):
    """Creates two layers with one workpiece each and returns them.

    Layer A holds wp_a (size 10x10 at origin); Layer B holds wp_b
    (size 10x10 at origin). Both are top-level and span two layers,
    which is the core scenario for the array tool.
    """
    layer_a = doc_editor.doc.active_layer
    layer_a.set_name("Layer A")
    wp_a = WorkPiece(name="wp_a")
    wp_a.set_size(10, 10)
    wp_a.pos = (0, 0)
    layer_a.add_child(wp_a)

    layer_b = Layer(name="Layer B")
    doc_editor.doc.add_child(layer_b)
    wp_b = WorkPiece(name="wp_b")
    wp_b.set_size(10, 10)
    wp_b.pos = (0, 0)
    layer_b.add_child(wp_b)

    return {
        "layer_a": layer_a,
        "layer_b": layer_b,
        "wp_a": wp_a,
        "wp_b": wp_b,
        "items": [wp_a, wp_b],
    }


def test_grid_array_creates_correct_copies(array_cmd, multilayer_selection):
    """3x2 pitch array of a 2-item multi-layer selection.

    6 instances total; the original is instance (0,0), so 5 copies are
    created. Each copy is one per source item -> 5 * 2 = 10 new items.
    """
    sel = multilayer_selection
    params = ArrayParams(
        mode=ArrayMode.GRID,
        grid=GridArrayParams(
            rows=2,
            cols=3,
            spacing_mode=SpacingMode.DISPLACEMENT,
            col_spacing_mm=20.0,
            row_spacing_mm=15.0,
        ),
    )

    created = array_cmd.create_array(sel["items"], params)
    assert len(created) == 10  # (6 - 1) instances * 2 source items

    # Each layer keeps its original plus 5 copies.
    a_items = sel["layer_a"].get_content_items()
    b_items = sel["layer_b"].get_content_items()
    assert len(a_items) == 6
    assert len(b_items) == 6

    # Copies must preserve layer membership: layer A only got wp_a copies.
    assert all(wp.name == "wp_a" for wp in a_items)
    assert all(wp.name == "wp_b" for wp in b_items)


def test_grid_pitch_positions(array_cmd, multilayer_selection):
    """Verify the world-space positions of a 1x2 pitch array."""
    sel = multilayer_selection
    params = ArrayParams(
        mode=ArrayMode.GRID,
        grid=GridArrayParams(
            rows=1,
            cols=2,
            spacing_mode=SpacingMode.DISPLACEMENT,
            col_spacing_mm=20.0,
            row_spacing_mm=0.0,
        ),
    )
    created = array_cmd.create_array(sel["items"], params)
    assert len(created) == 2

    # Original wp_a is at (0,0), size 10x10. Pitch 20 -> copy at (20,0).
    copy_a = next(c for c in created if c.name == "wp_a")
    assert copy_a.pos == pytest.approx((20.0, 0.0))
    # The copy keeps the source size.
    assert copy_a.size == pytest.approx((10.0, 10.0))


def test_gap_mode_spacing(array_cmd, multilayer_selection):
    """Gap mode: pitch = unit size + gap. Unit bbox is 10x10."""
    sel = multilayer_selection
    params = ArrayParams(
        mode=ArrayMode.GRID,
        grid=GridArrayParams(
            rows=1,
            cols=2,
            spacing_mode=SpacingMode.GAP,
            col_spacing_mm=5.0,
            row_spacing_mm=5.0,
        ),
    )
    created = array_cmd.create_array(sel["items"], params)
    copy_a = next(c for c in created if c.name == "wp_a")
    # pitch_x = unit_w(10) + gap(5) = 15 -> copy at (15, 0).
    assert copy_a.pos == pytest.approx((15.0, 0.0))


def test_original_is_kept_as_first_cell(array_cmd, multilayer_selection):
    """The original items are not duplicated/replaced; they stay put."""
    sel = multilayer_selection
    original_a_uid = sel["wp_a"].uid
    original_b_uid = sel["wp_b"].uid

    params = ArrayParams(grid=GridArrayParams(rows=2, cols=2))
    array_cmd.create_array(sel["items"], params)

    # Originals are still present, unmodified, at their original spot.
    assert sel["wp_a"].pos == pytest.approx((0.0, 0.0))
    assert sel["wp_a"].uid == original_a_uid
    assert sel["wp_b"].uid == original_b_uid
    # No created copy shares a uid with the originals.
    a_items = sel["layer_a"].get_content_items()
    assert all(item.uid != original_a_uid for item in a_items[1:])


def test_array_is_undoable(array_cmd, multilayer_selection):
    """Undo restores the document to exactly the two original items."""
    sel = multilayer_selection
    hm = array_cmd._editor.history_manager
    initial_len = len(hm.undo_stack)

    params = ArrayParams(grid=GridArrayParams(rows=2, cols=2))
    array_cmd.create_array(sel["items"], params)

    # One undo transaction was pushed.
    assert len(hm.undo_stack) == initial_len + 1
    assert len(sel["layer_a"].get_content_items()) == 4
    assert len(sel["layer_b"].get_content_items()) == 4

    hm.undo()

    # Back to one item per layer.
    assert len(sel["layer_a"].get_content_items()) == 1
    assert len(sel["layer_b"].get_content_items()) == 1
    assert sel["layer_a"].get_content_items()[0] is sel["wp_a"]


def test_circular_array_creates_copies(array_cmd, multilayer_selection):
    """A 4-copy full-circle circular array keeps the original + 3 copies."""
    sel = multilayer_selection
    params = ArrayParams(
        mode=ArrayMode.CIRCULAR,
        circular=CircularArrayParams(
            count=4,
            total_angle_deg=360.0,
            center_mm=(0.0, 0.0),
            rotate_copies=False,
        ),
    )
    created = array_cmd.create_array(sel["items"], params)
    # (4 - 1) instances * 2 source items.
    assert len(created) == 6


def test_point_rotation_array_creates_copies(array_cmd, multilayer_selection):
    """A 4-copy point-rotation array keeps the original + 3 copies.

    All copies share the original position; only their orientation
    differs.
    """
    sel = multilayer_selection
    params = ArrayParams(
        mode=ArrayMode.POINT_ROTATION,
        point_rotation=PointRotationParams(count=4, total_angle_deg=360.0),
    )
    created = array_cmd.create_array(sel["items"], params)
    # (4 - 1) rotated copies * 2 source items.
    assert len(created) == 6


def test_single_instance_is_noop(array_cmd, multilayer_selection):
    """A 1x1 array produces no copies at all."""
    sel = multilayer_selection
    params = ArrayParams(grid=GridArrayParams(rows=1, cols=1))
    created = array_cmd.create_array(sel["items"], params)
    assert created == []
    assert len(sel["layer_a"].get_content_items()) == 1


def test_compute_plan_is_pure(array_cmd, multilayer_selection):
    """compute_plan returns deltas without mutating the document."""
    sel = multilayer_selection
    before_count = len(sel["layer_a"].get_content_items())
    params = ArrayParams(grid=GridArrayParams(rows=2, cols=3))
    deltas = array_cmd.compute_plan(sel["items"], params)
    assert len(deltas) == 6
    assert deltas[0].is_identity()
    # No mutation happened.
    assert len(sel["layer_a"].get_content_items()) == before_count
