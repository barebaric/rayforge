import pytest
import numpy as np
from rayforge.core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_BEZIER,
    COL_TYPE,
    COL_Z,
)
from rayforge.core.geo.text import (
    text_to_geometry,
    get_font_metrics,
    get_text_width,
)


def test_text_to_geometry_empty_string():
    """Tests that empty string returns empty geometry."""
    geo = text_to_geometry("")
    assert geo.is_empty()
    assert geo.data is None


def test_text_to_whitespace_string():
    """Tests that whitespace-only string returns empty geometry."""
    geo = text_to_geometry("   ")
    assert geo.is_empty()


def test_text_to_geometry_basic():
    """Tests that basic text generates valid geometry."""
    geo = text_to_geometry("A")
    assert not geo.is_empty()
    assert geo.data is not None
    assert len(geo.data) > 0


def test_text_to_geometry_multiple_chars():
    """Tests that multiple characters generate geometry."""
    geo = text_to_geometry("ABC")
    assert not geo.is_empty()
    assert geo.data is not None
    assert len(geo.data) > 0


def test_text_to_geometry_has_move_to():
    """Tests that geometry starts with MOVE_TO command."""
    geo = text_to_geometry("X")
    assert geo.data is not None
    assert geo.data[0, COL_TYPE] == CMD_TYPE_MOVE


def test_text_to_geometry_has_path_commands():
    """Tests that geometry contains path commands."""
    geo = text_to_geometry("O")
    assert geo.data is not None
    types = set(geo.data[:, COL_TYPE])
    assert CMD_TYPE_MOVE in types
    assert CMD_TYPE_LINE in types or CMD_TYPE_BEZIER in types


def test_text_to_geometry_bezier_curves():
    """Tests that curved text generates bezier commands."""
    geo = text_to_geometry("O")
    assert geo.data is not None
    types = set(geo.data[:, COL_TYPE])
    assert CMD_TYPE_BEZIER in types


def test_text_to_geometry_font_family():
    """Tests that different font families generate geometry."""
    geo1 = text_to_geometry("A", font_family="sans-serif")
    geo2 = text_to_geometry("A", font_family="serif")
    assert not geo1.is_empty()
    assert not geo2.is_empty()
    assert geo1.data is not None
    assert geo2.data is not None


def test_text_to_geometry_font_size():
    """Tests that font size affects geometry scale."""
    geo_small = text_to_geometry("A", font_size=10.0)
    geo_large = text_to_geometry("A", font_size=20.0)
    assert not geo_small.is_empty()
    assert not geo_large.is_empty()

    rect_small = geo_small.rect()
    rect_large = geo_large.rect()

    width_small = rect_small[2] - rect_small[0]
    width_large = rect_large[2] - rect_large[0]

    assert width_large > width_small


def test_text_to_geometry_bold():
    """Tests that bold text generates geometry."""
    geo_normal = text_to_geometry("A", bold=False)
    geo_bold = text_to_geometry("A", bold=True)
    assert not geo_normal.is_empty()
    assert not geo_bold.is_empty()


def test_text_to_geometry_italic():
    """Tests that italic text generates geometry."""
    geo_normal = text_to_geometry("A", italic=False)
    geo_italic = text_to_geometry("A", italic=True)
    assert not geo_normal.is_empty()
    assert not geo_italic.is_empty()


def test_text_to_geometry_bold_italic():
    """Tests that bold italic text generates geometry."""
    geo = text_to_geometry("A", bold=True, italic=True)
    assert not geo.is_empty()
    assert geo.data is not None


def test_text_to_geometry_z_zero():
    """Tests that text geometry has z=0 for all points."""
    geo = text_to_geometry("ABC")
    assert geo.data is not None
    assert np.all(geo.data[:, COL_Z] == 0.0)


def test_text_to_geometry_consistent():
    """Tests that same text produces same geometry."""
    geo1 = text_to_geometry("TEST")
    geo2 = text_to_geometry("TEST")
    assert geo1.data is not None
    assert geo2.data is not None
    assert np.array_equal(geo1.data, geo2.data)


def test_text_to_geometry_different_text():
    """Tests that different text produces different geometry."""
    geo1 = text_to_geometry("A")
    geo2 = text_to_geometry("B")
    assert geo1.data is not None
    assert geo2.data is not None
    assert not np.array_equal(geo1.data, geo2.data)


def test_get_font_metrics_returns_tuple():
    """Tests that font metrics returns a tuple."""
    metrics = get_font_metrics()
    assert isinstance(metrics, tuple)
    assert len(metrics) == 3


def test_get_font_metrics_structure():
    """Tests that font metrics has (ascent, descent, height) structure."""
    ascent, descent, height = get_font_metrics()
    assert isinstance(ascent, float)
    assert isinstance(descent, float)
    assert isinstance(height, float)


def test_get_font_metrics_positive_height():
    """Tests that total height is positive."""
    _, _, height = get_font_metrics()
    assert height > 0


def test_get_font_metrics_descent_value():
    """Tests that descent has a numeric value."""
    _, descent, _ = get_font_metrics()
    assert isinstance(descent, float)


def test_get_font_metrics_height_valid():
    """Tests that height is a positive value."""
    _, _, height = get_font_metrics()
    assert height > 0


def test_get_font_metrics_font_size_scaling():
    """Tests that font size scales metrics approximately."""
    metrics_10 = get_font_metrics(font_size=10.0)
    metrics_20 = get_font_metrics(font_size=20.0)

    ascent_10, _, height_10 = metrics_10
    ascent_20, _, height_20 = metrics_20

    # Relaxed tolerance for CI environments where fonts may behave differently
    assert ascent_20 == pytest.approx(2 * ascent_10, rel=0.1)
    assert height_20 == pytest.approx(2 * height_10, rel=0.1)


def test_get_font_metrics_font_family():
    """Tests that font family parameter is accepted."""
    metrics_sans = get_font_metrics(font_family="sans-serif")
    metrics_serif = get_font_metrics(font_family="serif")

    assert len(metrics_sans) == 3
    assert len(metrics_serif) == 3


def test_get_font_metrics_bold():
    """Tests that bold font has metrics."""
    metrics = get_font_metrics(bold=True)
    ascent, descent, height = metrics
    assert height > 0


def test_get_font_metrics_italic():
    """Tests that italic font has metrics."""
    metrics = get_font_metrics(italic=True)
    ascent, descent, height = metrics
    assert height > 0


def test_get_text_width_empty_string():
    """Tests that empty string returns zero width."""
    width = get_text_width("")
    assert width == 0.0


def test_get_text_width_whitespace():
    """Tests that whitespace has positive width."""
    width = get_text_width(" ")
    assert width > 0.0


def test_get_text_width_single_char():
    """Tests that single character has positive width."""
    width = get_text_width("A")
    assert width > 0.0


def test_get_text_width_multiple_chars():
    """Tests that multiple characters have larger width."""
    width_single = get_text_width("A")
    width_multiple = get_text_width("AAA")
    assert width_multiple > width_single


def test_get_text_width_font_size_scaling():
    """Tests that font size scales width approximately."""
    width_10 = get_text_width("A", font_size=10.0)
    width_20 = get_text_width("A", font_size=20.0)
    assert width_20 == pytest.approx(2 * width_10, rel=0.1)


def test_get_text_width_different_chars():
    """Tests that different characters have different widths."""
    width_i = get_text_width("i")
    width_w = get_text_width("W")
    assert width_w > width_i


def test_get_text_width_bold():
    """Tests that bold text has width."""
    width = get_text_width("A", bold=True)
    assert width > 0.0


def test_get_text_width_italic():
    """Tests that italic text has width."""
    width = get_text_width("A", italic=True)
    assert width > 0.0


def test_get_text_width_font_family():
    """Tests that different font families have different widths."""
    # Use 'W' to maximize difference between proportional and fixed width fonts
    width_sans = get_text_width("W", font_family="sans-serif")
    width_mono = get_text_width("W", font_family="monospace")

    # In limited environments (CI), fonts might map to the same fallback.
    if width_sans == width_mono:
        pytest.skip(
            "System fonts for sans-serif and monospace appear identical"
        )

    assert width_sans != width_mono


def test_text_to_geometry_with_spaces():
    """Tests that text with spaces generates geometry."""
    geo = text_to_geometry("A B")
    assert not geo.is_empty()
    assert geo.data is not None


def test_text_to_geometry_special_chars():
    """Tests that special characters generate geometry."""
    geo = text_to_geometry("@#$")
    assert not geo.is_empty()
    assert geo.data is not None


def test_text_to_geometry_numbers():
    """Tests that numbers generate geometry."""
    geo = text_to_geometry("123")
    assert not geo.is_empty()
    assert geo.data is not None


def test_text_to_geometry_mixed_content():
    """Tests that mixed alphanumeric content generates geometry."""
    geo = text_to_geometry("Abc123!@#")
    assert not geo.is_empty()
    assert geo.data is not None


def test_text_to_geometry_small_font_size():
    """Tests that very small font size works."""
    geo = text_to_geometry("A", font_size=1.0)
    assert not geo.is_empty()


def test_text_to_geometry_large_font_size():
    """Tests that large font size works."""
    geo = text_to_geometry("A", font_size=100.0)
    assert not geo.is_empty()


def test_text_to_geometry_default_params():
    """Tests that default parameters work."""
    geo = text_to_geometry("A")
    assert not geo.is_empty()


def test_get_font_metrics_default_params():
    """Tests that default parameters work."""
    metrics = get_font_metrics()
    assert len(metrics) == 3


def test_get_text_width_default_params():
    """Tests that default parameters work."""
    width = get_text_width("A")
    assert width > 0.0
