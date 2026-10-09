import pytest

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.intent_builder import (
    job_design_rect,
    workpiece_world_aabb,
)


def _image_workpiece(pos, size) -> WorkPiece:
    """A workpiece without vector geometry, like an engraved image."""
    wp = WorkPiece(name="image")
    wp.set_size(*size)
    wp.pos = pos
    return wp


def test_aabb_of_workpiece_without_geometry():
    wp = _image_workpiece((27.2, 43.0), (85.6, 54.0))
    assert wp.get_world_geometry() is None
    assert workpiece_world_aabb(wp) == pytest.approx((27.2, 43.0, 112.8, 97.0))


def test_aabb_of_rotated_workpiece_without_geometry():
    wp = _image_workpiece((0.0, 0.0), (20.0, 10.0))
    wp.angle = 90.0
    min_x, min_y, max_x, max_y = workpiece_world_aabb(wp)
    assert (max_x - min_x, max_y - min_y) == pytest.approx((10.0, 20.0))


def test_design_rect_ignores_layers_without_visible_steps():
    doc = Doc()
    doc.active_layer.add_child(_image_workpiece((10, 10), (5, 5)))
    assert job_design_rect(doc) is None
