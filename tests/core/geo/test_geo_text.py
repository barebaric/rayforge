import pytest
from rayforge.core.geo.text import text_to_geometry
from rayforge.core.geo.constants import (
    CMD_TYPE_BEZIER,
    COL_TYPE,
)


def test_text_to_geometry_basic():
    """Test that generating text returns a valid, non-empty geometry."""
    geo = text_to_geometry("Test")
    assert not geo.is_empty()
    assert len(geo) > 0


def test_text_to_geometry_empty_string():
    """Test that an empty string returns empty geometry."""
    geo = text_to_geometry("")
    assert geo.is_empty()
    assert len(geo) == 0


def test_text_to_geometry_contains_curves():
    """
    Test that letters with curves (like 'O') generate Bezier commands.
    This ensures we are using copy_path() and not copy_path_flat().
    """
    geo = text_to_geometry("O", font_family="sans-serif")
    geo._sync_to_numpy()

    assert geo.data is not None
    cmd_types = geo.data[:, COL_TYPE]

    assert CMD_TYPE_BEZIER in cmd_types


def test_text_to_geometry_font_size():
    """
    Test that changing font size affects the resulting geometry dimensions.
    """
    # Create text with size 10
    geo_small = text_to_geometry("I", font_size=10.0)

    # Create text with size 20
    geo_large = text_to_geometry("I", font_size=20.0)

    # Get bounds
    min_x_s, min_y_s, max_x_s, max_y_s = geo_small.rect()
    min_x_l, min_y_l, max_x_l, max_y_l = geo_large.rect()

    height_small = max_y_s - min_y_s
    height_large = max_y_l - min_y_l

    # The large text should be roughly twice as tall.
    # We use a relaxed tolerance (0.2) because Cairo/Freetype scaling of
    # outlines is not always perfectly linear regarding bounding box
    # due to hinting or internal metrics.
    assert height_large > height_small
    assert height_large == pytest.approx(height_small * 2.0, rel=0.2)


def test_text_to_geometry_path_continuity():
    """Test that the generated path forms closed loops (for letters)."""
    geo = text_to_geometry("O")

    # 'O' usually consists of two closed contours (outer and inner)
    contours = geo.split_into_contours()
    assert len(contours) >= 1

    for contour in contours:
        # Letters should form closed paths.
        # Note: If this fails, it usually means text_to_geometry didn't
        # filter out dangling MOVE commands correctly (e.g. cursor advance).
        assert contour.is_closed(tolerance=1e-3)
