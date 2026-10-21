"""Tests for hit-testing perfectly flat canvas elements."""

from typing import TYPE_CHECKING, cast

from raygeo.geo import Matrix

from rayforge.ui_gtk.canvas.element import CanvasElement

if TYPE_CHECKING:
    from rayforge.ui_gtk.canvas.canvas import Canvas


class FakeCanvas:
    def __init__(self):
        self.view_transform = Matrix.identity()

    def queue_draw(self):
        pass


def _make_element(matrix, width=1.0, height=1.0, pixel_perfect=True):
    """Builds a buffered element with the given transform."""
    element = CanvasElement(
        x=0.0,
        y=0.0,
        width=width,
        height=height,
        canvas=cast("Canvas", FakeCanvas()),
        buffered=True,
        pixel_perfect_hit=pixel_perfect,
        hit_distance=5,
    )
    element.set_transform(matrix)
    return element


class TestFlatElementHit:
    """Flat elements must be selectable within a small tolerance."""

    def test_horizontal_line_is_hit_on_the_line(self):
        element = _make_element(
            Matrix.from_list(
                [[253.98, 0.0, 22.61], [0.0, 1.0325e-09, 18.9384], [0, 0, 1]]
            )
        )
        assert element.get_elem_hit(100.0, 18.9384, selectable=True) is element

    def test_horizontal_line_is_hit_near_the_line(self):
        element = _make_element(
            Matrix.from_list(
                [[253.98, 0.0, 22.61], [0.0, 1.0325e-09, 18.9384], [0, 0, 1]]
            )
        )
        assert element.get_elem_hit(100.0, 21.0, selectable=True) is element

    def test_horizontal_line_misses_far_above(self):
        element = _make_element(
            Matrix.from_list(
                [[253.98, 0.0, 22.61], [0.0, 1.0325e-09, 18.9384], [0, 0, 1]]
            )
        )
        assert element.get_elem_hit(100.0, 40.0, selectable=True) is None

    def test_vertical_line_is_hit_near_the_line(self):
        element = _make_element(
            Matrix.from_list(
                [[1.0325e-09, 0.0, 18.9384], [0.0, 253.98, 22.61], [0, 0, 1]]
            )
        )
        assert element.get_elem_hit(21.0, 100.0, selectable=True) is element

    def test_zero_height_element_is_hit_near_the_line(self):
        element = _make_element(
            Matrix.translation(10.0, 20.0), width=200.0, height=0.0
        )
        assert element.get_elem_hit(50.0, 21.0, selectable=True) is element
        assert element.get_elem_hit(50.0, 26.0, selectable=True) is None


class TestNormalElementHit:
    """Non-flat elements keep their exact bounding box."""

    def test_click_just_outside_bounding_box_misses(self):
        element = _make_element(
            Matrix.identity(), width=100.0, height=100.0, pixel_perfect=False
        )
        assert element.get_elem_hit(100.5, 50.0, selectable=True) is None
        assert element.get_elem_hit(50.0, 100.5, selectable=True) is None

    def test_click_inside_bounding_box_hits(self):
        element = _make_element(
            Matrix.identity(), width=100.0, height=100.0, pixel_perfect=False
        )
        assert element.get_elem_hit(50.0, 50.0, selectable=True) is element
