import pytest

from rayforge.shared.units.engine import engine


def test_convert_between_imperial_length_units():
    """Registered imperial length units convert directly between each other."""
    inches_result, inches_unit = engine.convert(12.0, "in", "ft")
    feet_result, feet_unit = engine.convert(1.0, "ft", "in")

    assert inches_result == pytest.approx(1.0)
    assert inches_unit == "ft"
    assert feet_result == pytest.approx(12.0)
    assert feet_unit == "in"
