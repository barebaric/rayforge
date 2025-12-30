import pytest
from unittest.mock import Mock
from rayforge.core.sketcher.tools.base import SketchTool


@pytest.fixture
def mock_element():
    """Create a mock SketchElement for testing."""
    element = Mock()
    return element


@pytest.fixture
def sketch_tool(mock_element):
    """Create a concrete SketchTool subclass for testing."""

    class ConcreteSketchTool(SketchTool):
        def on_press(self, world_x, world_y, n_press):
            return False

        def on_drag(self, world_dx, world_dy):
            pass

        def on_release(self, world_x, world_y):
            pass

    return ConcreteSketchTool(mock_element)


def test_sketch_tool_initialization(sketch_tool, mock_element):
    """Test that SketchTool initializes correctly."""
    assert sketch_tool.element == mock_element


def test_sketch_tool_on_hover_motion_default(sketch_tool):
    """Test that on_hover_motion does nothing by default."""
    sketch_tool.on_hover_motion(100.0, 200.0)
    assert True


def test_sketch_tool_on_deactivate_default(sketch_tool):
    """Test that on_deactivate does nothing by default."""
    sketch_tool.on_deactivate()
    assert True


def test_sketch_tool_draw_overlay_default(sketch_tool):
    """Test that draw_overlay does nothing by default."""
    import cairo

    ctx = Mock(spec=cairo.Context)
    sketch_tool.draw_overlay(ctx)
    assert True
