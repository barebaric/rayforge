from rayforge.core.ops.axis import Axis
from rayforge.ui_gtk.canvas2d.projection import CanvasProjection


def test_default_projection():
    p = CanvasProjection()
    assert p.horizontal_axis == Axis.X
    assert p.vertical_axis == Axis.Y


def test_default_circumferential_axis():
    p = CanvasProjection()
    assert p.circumferential_axis == Axis.Y


def test_projection_is_frozen():
    p = CanvasProjection()
    try:
        p.horizontal_axis = Axis.Y  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_projection_equality():
    a = CanvasProjection()
    b = CanvasProjection(Axis.X, Axis.Y)
    assert a == b


def test_projection_inequality():
    a = CanvasProjection()
    b = CanvasProjection(Axis.Y, Axis.X)
    assert a != b
