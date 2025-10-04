from rayforge.shared.units.definitions import (
    get_unit,
    get_units_for_quantity,
    get_base_unit_for_quantity,
)
from rayforge.shared.units.engine import engine
from rayforge.core.config import Config


def test_length_units():
    """Test that length units are properly defined and convertible."""
    print("Testing length units...")

    # Test that length units are registered
    units = get_units_for_quantity("length")
    print(f"Found {len(units)} length units")

    # Test that we have the expected units
    expected_units = ["mm", "cm", "m", "in", "ft"]
    unit_names = [u.name for u in units]

    for expected in expected_units:
        assert expected in unit_names, f"Expected unit {expected} not found"

    # Test base unit
    base_unit = get_base_unit_for_quantity("length")
    assert base_unit is not None, "No base unit found for length"
    assert base_unit.name == "mm", (
        f"Expected base unit mm, got {base_unit.name}"
    )

    # Test unit conversions
    test_cases = [
        (1000.0, "mm", "cm", 100.0),
        (1000.0, "mm", "m", 1.0),
        (25.4, "mm", "in", 1.0),
        (1.0, "m", "mm", 1000.0),
        (1.0, "in", "mm", 25.4),
        (1.0, "ft", "mm", 304.8),
    ]

    for value, from_unit, to_unit, expected in test_cases:
        result, _ = engine.convert(value, from_unit, to_unit)
        assert abs(result - expected) < 0.01, (
            f"Conversion failed: {value} {from_unit} -> {to_unit}, "
            f"expected {expected}, got {result}"
        )

    print("✓ Length units test passed")


def test_speed_units():
    """Test that speed units are properly defined and convertible."""
    print("Testing speed units...")

    # Test that speed units are registered
    units = get_units_for_quantity("speed")
    print(f"Found {len(units)} speed units")

    # Test that we have the expected units
    expected_units = ["mm/min", "mm/s", "in/min", "in/s"]
    unit_names = [u.name for u in units]

    for expected in expected_units:
        assert expected in unit_names, f"Expected unit {expected} not found"

    # Test base unit
    base_unit = get_base_unit_for_quantity("speed")
    assert base_unit is not None, "No base unit found for speed"
    assert base_unit.name == "mm/min", (
        f"Expected base unit mm/min, got {base_unit.name}"
    )

    # Test unit conversions
    test_cases = [
        (60.0, "mm/s", "mm/min", 3600.0),
        (1.0, "mm/min", "mm/s", 0.01667),
        (25.4, "mm/min", "in/min", 1.0),
        (1.0, "in/min", "mm/min", 25.4),
    ]

    for value, from_unit, to_unit, expected in test_cases:
        result, _ = engine.convert(value, from_unit, to_unit)
        assert abs(result - expected) < 0.01, (
            f"Conversion failed: {value} {from_unit} -> {to_unit}, "
            f"expected {expected}, got {result}"
        )

    print("✓ Speed units test passed")


def test_acceleration_units():
    """Test that acceleration units are properly defined and convertible."""
    print("Testing acceleration units...")

    # Test that acceleration units are registered
    units = get_units_for_quantity("acceleration")
    print(f"Found {len(units)} acceleration units")

    # Test that we have the expected units
    expected_units = ["mm/s²", "cm/s²", "m/s²", "in/s²", "ft/s²"]
    unit_names = [u.name for u in units]

    for expected in expected_units:
        assert expected in unit_names, f"Expected unit {expected} not found"

    # Test base unit
    base_unit = get_base_unit_for_quantity("acceleration")
    assert base_unit is not None, "No base unit found for acceleration"
    assert base_unit.name == "mm/s²", (
        f"Expected base unit mm/s², got {base_unit.name}"
    )

    # Test unit conversions
    test_cases = [
        (1000, "mm/s²", "cm/s²", 100.0),
        (1000, "mm/s²", "m/s²", 1.0),
        (1000, "mm/s²", "in/s²", 39.37),
        (1.0, "m/s²", "mm/s²", 1000.0),
        (1.0, "in/s²", "mm/s²", 25.4),
    ]

    for value, from_unit, to_unit, expected in test_cases:
        result, _ = engine.convert(value, from_unit, to_unit)
        assert abs(result - expected) < 0.01, (
            f"Conversion failed: {value} {from_unit} -> {to_unit}, "
            f"expected {expected}, got {result}"
        )

    print("✓ Acceleration units test passed")


def test_config_integration():
    """Test that unit preferences are properly integrated with the config."""
    print("Testing config integration...")

    # Test config integration
    config = Config()

    # Test that all quantities are in unit preferences
    assert "length" in config.unit_preferences, (
        "Length not in unit preferences"
    )
    assert "speed" in config.unit_preferences, "Speed not in unit preferences"
    assert "acceleration" in config.unit_preferences, (
        "Acceleration not in unit preferences"
    )

    # Test default values
    assert config.unit_preferences["length"] == "mm", (
        f"Expected mm, got {config.unit_preferences['length']}"
    )
    assert config.unit_preferences["speed"] == "mm/min", (
        f"Expected mm/min, got {config.unit_preferences['speed']}"
    )
    assert config.unit_preferences["acceleration"] == "mm/s²", (
        f"Expected mm/s², got {config.unit_preferences['acceleration']}"
    )

    print("✓ Config integration test passed")


def test_unit_conversion_methods():
    """Test the unit conversion methods on Unit objects."""
    print("Testing unit conversion methods...")

    # Test length units
    mm_unit = get_unit("mm")
    cm_unit = get_unit("cm")

    assert mm_unit is not None, "mm unit not found"
    assert cm_unit is not None, "cm unit not found"

    # Test to_base conversion
    value_in_cm = 10.0  # 10 cm
    value_in_mm = cm_unit.to_base(value_in_cm)
    assert abs(value_in_mm - 100.0) < 0.01, (
        f"Base conversion failed: expected 100, got {value_in_mm}"
    )

    # Test from_base conversion
    value_in_mm = 100.0  # 100 mm (already in base units)
    value_in_cm = cm_unit.from_base(value_in_mm)
    assert abs(value_in_cm - 10.0) < 0.01, (
        f"From base conversion failed: expected 10, got {value_in_cm}"
    )

    # Test acceleration units
    mm_acc_unit = get_unit("mm/s²")
    cm_acc_unit = get_unit("cm/s²")

    assert mm_acc_unit is not None, "mm/s² unit not found"
    assert cm_acc_unit is not None, "cm/s² unit not found"

    # Test to_base conversion for acceleration
    value_in_cm_acc = 100.0  # 100 cm/s²
    value_in_mm_acc = cm_acc_unit.to_base(value_in_cm_acc)
    assert abs(value_in_mm_acc - 1000.0) < 0.01, (
        f"Base conversion failed: expected 1000, got {value_in_mm_acc}"
    )

    print("✓ Unit conversion methods test passed")


def test_engine_normalization():
    """Test the engine's unit symbol normalization."""
    print("Testing engine normalization...")

    # Test length unit normalization
    assert engine.normalize_unit_symbol("millimeter") == "mm"
    assert engine.normalize_unit_symbol("inches") == "in"
    assert engine.normalize_unit_symbol("feet") == "ft"

    # Test compound unit normalization
    assert engine.normalize_unit_symbol("mm/second") == "mm/s"
    assert engine.normalize_unit_symbol("in/min") == "in/min"

    print("✓ Engine normalization test passed")


def test_all():
    """Run all unit tests."""
    print("Running comprehensive unit tests...\n")

    test_length_units()
    test_speed_units()
    test_acceleration_units()
    test_config_integration()
    test_unit_conversion_methods()
    test_engine_normalization()

    print("\nAll tests passed! ✅")


if __name__ == "__main__":
    test_all()
