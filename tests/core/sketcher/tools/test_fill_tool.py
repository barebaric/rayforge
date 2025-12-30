import pytest
from unittest.mock import Mock
from rayforge.core.sketcher.tools.fill_tool import FillTool


@pytest.fixture
def mock_element():
    """Create a mock SketchElement for testing."""
    element = Mock()
    element.sketch = Mock()
    element.sketch.registry = Mock()
    element.sketch.fills = []
    element.sketch._find_all_closed_loops = Mock(return_value=[])
    element.sketch._calculate_loop_signed_area = Mock(return_value=100.0)
    element.hittester = Mock()
    element.hittester.screen_to_model = Mock(return_value=(0.0, 0.0))
    element.hittester.get_hit_data = Mock(return_value=(None, None))
    element.mark_dirty = Mock()
    element.editor = None
    return element


@pytest.fixture
def fill_tool(mock_element):
    """Create a FillTool instance for testing."""
    return FillTool(mock_element)


def test_fill_tool_initialization(fill_tool, mock_element):
    """Test that FillTool initializes correctly."""
    assert fill_tool.element == mock_element


def test_fill_tool_on_press_no_loops(fill_tool, mock_element):
    """Test on_press when no loops are found."""
    mock_element.sketch._find_all_closed_loops.return_value = []

    result = fill_tool.on_press(100.0, 200.0, 1)

    assert result is False


def test_fill_tool_on_press_double_click(fill_tool):
    """Test on_press with double click (n_press != 1)."""
    result = fill_tool.on_press(100.0, 200.0, 2)

    assert result is False


def test_fill_tool_on_drag(fill_tool):
    """Test on_drag does nothing."""
    fill_tool.on_drag(10.0, 20.0)
    assert True


def test_fill_tool_on_release(fill_tool):
    """Test on_release does nothing."""
    fill_tool.on_release(10.0, 20.0)
    assert True


def test_fill_tool_on_press_with_loop(fill_tool, mock_element):
    """Test on_press when a loop is found."""
    mock_element.sketch._find_all_closed_loops.return_value = []
    mock_element.hittester.screen_to_model.return_value = (50.0, 50.0)

    result = fill_tool.on_press(100.0, 200.0, 1)

    assert result is False


def test_fill_tool_on_press_existing_fill(fill_tool, mock_element):
    """Test on_press when a fill already exists."""
    from rayforge.core.sketcher.sketch import Fill

    existing_fill = Fill(uid="test", boundary=[(1, True), (2, True)])
    mock_element.sketch.fills = [existing_fill]
    mock_element.sketch._find_all_closed_loops.return_value = []
    mock_element.hittester.screen_to_model.return_value = (50.0, 50.0)

    result = fill_tool.on_press(100.0, 200.0, 1)

    assert result is False
