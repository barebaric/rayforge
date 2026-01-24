import numpy as np
from rayforge.core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_BEZIER,
    COL_TYPE,
    COL_Z,
)
from rayforge.core.geo.font_config import FontConfig
from rayforge.core.geo.text import text_to_geometry


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
    config1 = FontConfig(font_family="sans-serif")
    config2 = FontConfig(font_family="serif")
    geo1 = text_to_geometry("A", font_config=config1)
    geo2 = text_to_geometry("A", font_config=config2)
    assert not geo1.is_empty()
    assert not geo2.is_empty()
    assert geo1.data is not None
    assert geo2.data is not None


def test_text_to_geometry_font_size():
    """Tests that font size affects geometry scale."""
    config_small = FontConfig(font_size=10.0)
    config_large = FontConfig(font_size=20.0)
    geo_small = text_to_geometry("A", font_config=config_small)
    geo_large = text_to_geometry("A", font_config=config_large)
    assert not geo_small.is_empty()
    assert not geo_large.is_empty()

    rect_small = geo_small.rect()
    rect_large = geo_large.rect()

    width_small = rect_small[2] - rect_small[0]
    width_large = rect_large[2] - rect_large[0]

    assert width_large > width_small


def test_text_to_geometry_bold():
    """Tests that bold text generates geometry."""
    config_normal = FontConfig(bold=False)
    config_bold = FontConfig(bold=True)
    geo_normal = text_to_geometry("A", font_config=config_normal)
    geo_bold = text_to_geometry("A", font_config=config_bold)
    assert not geo_normal.is_empty()
    assert not geo_bold.is_empty()


def test_text_to_geometry_italic():
    """Tests that italic text generates geometry."""
    config_normal = FontConfig(italic=False)
    config_italic = FontConfig(italic=True)
    geo_normal = text_to_geometry("A", font_config=config_normal)
    geo_italic = text_to_geometry("A", font_config=config_italic)
    assert not geo_normal.is_empty()
    assert not geo_italic.is_empty()


def test_text_to_geometry_bold_italic():
    """Tests that bold italic text generates geometry."""
    config = FontConfig(bold=True, italic=True)
    geo = text_to_geometry("A", font_config=config)
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
    config = FontConfig(font_size=1.0)
    geo = text_to_geometry("A", font_config=config)
    assert not geo.is_empty()


def test_text_to_geometry_large_font_size():
    """Tests that large font size works."""
    config = FontConfig(font_size=100.0)
    geo = text_to_geometry("A", font_config=config)
    assert not geo.is_empty()


def test_text_to_geometry_default_params():
    """Tests that default parameters work."""
    geo = text_to_geometry("A")
    assert not geo.is_empty()
