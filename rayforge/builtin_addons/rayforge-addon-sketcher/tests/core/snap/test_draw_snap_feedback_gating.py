"""
draw_snap_feedback must not render a snap result whose geometry has
moved since the query: during fast drags the solve and array sync run
between the snap query and the frame draw, and the stale result
flashes snap lines that reference no visible geometry.
"""

from typing import Any, cast

import cairo
import pytest
from sketcher.core.snap.types import SnapResult
from sketcher.ui_gtk.tools.snap_mixin import SnapMixin


class _StubEngine:
    def __init__(self, current: bool):
        self._current = current

    def is_snap_result_current(self, registry) -> bool:
        return self._current


class _StubHittester:
    @staticmethod
    def get_model_to_screen_transform(element):
        raise AssertionError("stale snap result must not be drawn")


class _StubSketch:
    registry = object()


class _StubCanvas:
    pass


class _StubElement:
    canvas = _StubCanvas()
    sketch = _StubSketch()
    hittester = _StubHittester()

    def __init__(self, engine: _StubEngine):
        self.snap_engine = engine


class _DrawTool(SnapMixin):
    pass


@pytest.fixture
def surface_ctx():
    surface = cairo.ImageSurface(cairo.Format.ARGB32, 4, 4)
    return cairo.Context(surface)


def test_draw_skips_stale_snap_result(surface_ctx):
    element = _StubElement(_StubEngine(current=False))
    tool = _DrawTool()
    tool.current_snap_result = SnapResult(snapped=True, position=(0.0, 0.0))

    tool.draw_snap_feedback(surface_ctx, cast(Any, element))


def test_draw_renders_current_snap_result(surface_ctx):
    element = _StubElement(_StubEngine(current=True))
    tool = _DrawTool()
    tool.current_snap_result = SnapResult(snapped=True, position=(0.0, 0.0))

    with pytest.raises(AssertionError):
        tool.draw_snap_feedback(surface_ctx, cast(Any, element))


def test_draw_skips_without_snap_result(surface_ctx):
    element = _StubElement(_StubEngine(current=True))
    tool = _DrawTool()
    tool.current_snap_result = None

    tool.draw_snap_feedback(surface_ctx, cast(Any, element))
