from unittest.mock import MagicMock

from rayforge.core.capability import MachineCapability
from rayforge.machine.driver.driver import PWMParams
from rayforge.machine.models.head import Head
from rayforge.machine.models.laser import (
    DEFAULT_FOCAL_DISTANCE_MM,
    DEFAULT_MAX_POWER_WATTS,
    Laser,
    LaserHead,
    LaserType,
    effective_focal_distance,
)


def test_laser_initialization():
    """Test that a new laser initializes with default values."""
    laser = Laser()

    assert laser.name == "Laser Head"
    assert laser.tool_number == 0
    assert laser.max_power == 1000
    assert laser.frame_power_percent == 0
    assert laser.focus_power_percent == 0
    assert laser.spot_size_mm == (0.1, 0.1)
    assert laser.uid is not None


def test_effective_focal_distance_uses_configured_value():
    laser = LaserHead()
    laser.focal_distance = 30.0
    assert effective_focal_distance(laser) == 30.0


def test_effective_focal_distance_defaults_for_laser_head():
    laser = LaserHead()
    laser.focal_distance = 0.0
    assert effective_focal_distance(laser) == DEFAULT_FOCAL_DISTANCE_MM


def test_effective_focal_distance_zero_for_non_laser():
    assert effective_focal_distance(Head()) == 0.0
    assert effective_focal_distance(None) == 0.0


def test_set_focus_power():
    """
    Test that set_focus_power updates the focus power percent and sends
    signal.
    """
    laser = Laser()

    signal_calls = []

    def signal_handler(sender):
        signal_calls.append(sender)

    laser.changed.connect(signal_handler)

    laser.set_focus_power(0.05)

    assert laser.focus_power_percent == 0.05
    assert len(signal_calls) == 1
    assert signal_calls[0] is laser


def test_set_focus_power_zero():
    """Test that setting focus power to 0 works correctly."""
    laser = Laser()

    laser.set_focus_power(0.0)

    assert laser.focus_power_percent == 0.0


def test_set_focus_power_multiple_values():
    """Test setting focus power to various values."""
    laser = Laser()

    test_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    for value in test_values:
        laser.set_focus_power(value)
        assert laser.focus_power_percent == value


def test_laser_serialization_includes_focus_power():
    """Test that focus power is included in serialization."""
    laser = Laser()
    laser.set_focus_power(0.42)

    data = laser.to_dict()

    assert "focus_power_percent" in data
    assert data["focus_power_percent"] == 42
    assert "focus_power" not in data


def test_laser_deserialization_focus_power():
    """Test that focus power is correctly deserialized."""
    data = {
        "uid": "test-uid",
        "name": "Test Laser",
        "tool_number": 1,
        "max_power": 1500,
        "frame_power": 20,
        "focus_power": 35,
        "spot_size_mm": [0.15, 0.15],
    }

    laser = Laser.from_dict(data)

    assert laser.focus_power_percent == 35 / 1500
    assert laser.frame_power_percent == 20 / 1500

    assert laser.uid == "test-uid"
    assert laser.name == "Test Laser"
    assert laser.tool_number == 1
    assert laser.max_power == 1500
    assert laser.spot_size_mm == [0.15, 0.15]


def test_laser_deserialization_default_focus_power():
    """Test that focus power defaults to 0 when not in data."""
    data = {
        "uid": "test-uid",
        "name": "Test Laser",
        "tool_number": 1,
        "max_power": 1500,
        "frame_power": 20,
        "spot_size_mm": [0.15, 0.15],
    }

    laser = Laser.from_dict(data)

    assert laser.focus_power_percent == 0


def test_laser_roundtrip_serialization():
    """Test that focus power survives a full serialization roundtrip."""
    original_laser = Laser()
    original_laser.set_focus_power(0.67)
    original_laser.name = "Roundtrip Test"

    data = original_laser.to_dict()

    new_laser = Laser.from_dict(data)

    assert new_laser.focus_power_percent == original_laser.focus_power_percent
    assert new_laser.name == original_laser.name
    assert new_laser.uid == original_laser.uid


def test_backward_compatibility_old_format():
    """Test that old format data (without _percent fields) is converted
    correctly.
    """
    old_data = {
        "uid": "old-uid",
        "name": "Old Laser",
        "tool_number": 1,
        "max_power": 1000,
        "frame_power": 100,
        "focus_power": 500,
        "spot_size_mm": [0.2, 0.2],
    }

    laser = Laser.from_dict(old_data)

    assert laser.frame_power_percent == 0.1
    assert laser.focus_power_percent == 0.5


def test_new_format_deserialization():
    """Test that new format data (with _percent fields) is deserialized
    correctly.
    """
    new_data = {
        "uid": "new-uid",
        "name": "New Laser",
        "tool_number": 2,
        "max_power": 2000,
        "frame_power": 400,
        "frame_power_percent": 0.2,
        "focus_power": 1000,
        "focus_power_percent": 0.5,
        "spot_size_mm": [0.3, 0.3],
    }

    laser = Laser.from_dict(new_data)

    assert laser.frame_power_percent == 0.002
    assert laser.focus_power_percent == 0.005


def test_serialization_includes_percent_fields():
    """Test that serialization includes only percentage fields."""
    laser = Laser()
    laser.set_frame_power(0.3)
    laser.set_focus_power(0.75)

    data = laser.to_dict()

    assert "frame_power_percent" in data
    assert "focus_power_percent" in data
    assert data["frame_power_percent"] == 30
    assert data["focus_power_percent"] == 75

    assert "frame_power" not in data
    assert "focus_power" not in data


def test_laser_forward_compatibility_with_extra_fields():
    """
    Tests that from_dict() preserves extra fields from newer versions
    and to_dict() re-serializes them.
    """
    laser_dict = {
        "uid": "laser-forward-456",
        "name": "Future Laser",
        "tool_number": 1,
        "max_power": 1500,
        "frame_power_percent": 20,
        "focus_power_percent": 35,
        "spot_size_mm": [0.15, 0.15],
        "future_field_string": "some value",
        "future_field_number": 42,
        "future_field_dict": {"nested": "data"},
    }

    laser = Laser.from_dict(laser_dict)

    assert laser.extra["future_field_string"] == "some value"
    assert laser.extra["future_field_number"] == 42
    assert laser.extra["future_field_dict"] == {"nested": "data"}

    data = laser.to_dict()
    assert data["future_field_string"] == "some value"
    assert data["future_field_number"] == 42
    assert data["future_field_dict"] == {"nested": "data"}


def test_laser_backward_compat_missing_optional_fields():
    """
    Tests that from_dict() handles missing optional fields gracefully
    (simulating data from an older version).
    """
    minimal_dict = {
        "uid": "laser-backward-789",
        "name": "Old Laser",
        "tool_number": 0,
        "max_power": 1000,
    }

    laser = Laser.from_dict(minimal_dict)

    assert laser.name == "Old Laser"
    assert laser.frame_power_percent == 0
    assert laser.focus_power_percent == 0
    assert laser.spot_size_mm == (0.1, 0.1)
    assert laser.extra == {}


def test_pwm_defaults():
    """PWM attributes have reasonable defaults."""
    laser = Laser()
    assert laser.pwm_frequency == 500
    assert laser.max_pwm_frequency == 5000
    assert laser.pulse_width == 50
    assert laser.min_pulse_width == 5
    assert laser.max_pulse_width == 500


def test_set_pwm_frequency():
    laser = Laser()
    signal_calls = []

    def handler(sender):
        signal_calls.append(sender)

    laser.changed.connect(handler)
    laser.set_pwm_frequency(1000)
    assert laser.pwm_frequency == 1000
    assert len(signal_calls) == 1


def test_set_pwm_frequency_clamped_by_max():
    laser = Laser()
    laser.set_max_pwm_frequency(2000)
    laser.set_pwm_frequency(5000)
    assert laser.pwm_frequency == 2000


def test_set_max_pwm_frequency_clamps_frequency():
    laser = Laser()
    laser.set_pwm_frequency(4000)
    laser.set_max_pwm_frequency(2000)
    assert laser.pwm_frequency == 2000


def test_set_max_pwm_frequency():
    laser = Laser()
    signal_calls = []

    def handler(sender):
        signal_calls.append(sender)

    laser.changed.connect(handler)
    laser.set_max_pwm_frequency(10000)
    assert laser.max_pwm_frequency == 10000
    assert len(signal_calls) == 1


def test_set_pulse_width():
    laser = Laser()
    signal_calls = []

    def handler(sender):
        signal_calls.append(sender)

    laser.changed.connect(handler)
    laser.set_pulse_width(100)
    assert laser.pulse_width == 100
    assert len(signal_calls) == 1


def test_set_pulse_width_clamped_to_bounds():
    laser = Laser()
    laser.set_min_pulse_width(10)
    laser.set_max_pulse_width(200)
    laser.set_pulse_width(5)
    assert laser.pulse_width == 10
    laser.set_pulse_width(500)
    assert laser.pulse_width == 200


def test_set_min_pulse_width():
    laser = Laser()
    signal_calls = []

    def handler(sender):
        signal_calls.append(sender)

    laser.changed.connect(handler)
    laser.set_min_pulse_width(10)
    assert laser.min_pulse_width == 10
    assert len(signal_calls) == 1


def test_set_min_pulse_width_pushes_max_and_pulse_width():
    laser = Laser()
    laser.set_max_pulse_width(100)
    laser.set_pulse_width(50)
    laser.set_min_pulse_width(200)
    assert laser.max_pulse_width == 200
    assert laser.pulse_width == 200


def test_set_max_pulse_width():
    laser = Laser()
    signal_calls = []

    def handler(sender):
        signal_calls.append(sender)

    laser.changed.connect(handler)
    laser.set_max_pulse_width(100)
    assert laser.max_pulse_width == 100
    assert len(signal_calls) == 1


def test_set_max_pulse_width_pushes_min_and_pulse_width():
    laser = Laser()
    laser.set_min_pulse_width(10)
    laser.set_pulse_width(50)
    laser.set_max_pulse_width(5)
    assert laser.min_pulse_width == 5
    assert laser.pulse_width == 5


def test_setters_no_signal_on_same_value():
    laser = Laser()
    signal_calls = []

    def handler(sender):
        signal_calls.append(sender)

    laser.changed.connect(handler)
    laser.set_pwm_frequency(500)
    laser.set_max_pwm_frequency(5000)
    laser.set_pulse_width(50)
    laser.set_min_pulse_width(5)
    laser.set_max_pulse_width(500)
    assert len(signal_calls) == 0


def test_pwm_serialization_roundtrip():
    laser = Laser()
    laser.set_pwm_frequency(1000)
    laser.set_max_pwm_frequency(5000)
    laser.set_pulse_width(50)
    laser.set_min_pulse_width(1)
    laser.set_max_pulse_width(100)

    data = laser.to_dict()
    assert data["pwm_frequency"] == 1000
    assert data["max_pwm_frequency"] == 5000
    assert data["pulse_width"] == 50
    assert data["min_pulse_width"] == 1
    assert data["max_pulse_width"] == 100

    restored = Laser.from_dict(data)
    assert restored.pwm_frequency == 1000
    assert restored.max_pwm_frequency == 5000
    assert restored.pulse_width == 50
    assert restored.min_pulse_width == 1
    assert restored.max_pulse_width == 100


def test_pwm_missing_fields_use_init_defaults():
    data = {
        "uid": "test-uid",
        "name": "Test Laser",
        "tool_number": 0,
        "max_power": 1000,
    }
    laser = Laser.from_dict(data)
    assert laser.pwm_frequency == 500
    assert laser.max_pwm_frequency == 5000
    assert laser.pulse_width == 50
    assert laser.min_pulse_width == 5
    assert laser.max_pulse_width == 500


def test_supports_pwm_delegates_to_driver(isolated_machine):
    """Machine.supports_pwm delegates to the driver."""
    head = LaserHead()
    isolated_machine.heads = [head]
    mock_driver = isolated_machine.driver
    mock_driver.supports_pwm = MagicMock(return_value=False)

    assert isolated_machine.supports_pwm(head) is False
    mock_driver.supports_pwm.assert_called_once_with(head)


def test_get_capabilities_includes_pwm_when_driver_supports_it(
    isolated_machine,
):
    """A PWM machine capability is inferred from driver support."""
    head = LaserHead()
    isolated_machine.heads = [head]
    mock_driver = isolated_machine.driver
    mock_driver.supports_pwm = MagicMock(return_value=False)

    assert MachineCapability.PWM not in isolated_machine.get_capabilities()

    mock_driver.supports_pwm = MagicMock(return_value=True)
    assert MachineCapability.PWM in isolated_machine.get_capabilities()


def test_get_pwm_params_returns_driver_params(isolated_machine):
    """Machine.get_pwm_params returns the driver-reported params."""
    head = LaserHead()
    isolated_machine.heads = [head]
    mock_driver = isolated_machine.driver
    params = PWMParams(1000, 5000, 50, 5, 500)
    mock_driver.get_pwm_params = MagicMock(return_value=params)

    assert isolated_machine.get_pwm_params(head) is params


def test_get_pwm_settings_none_without_pwm(isolated_machine):
    head = LaserHead()
    isolated_machine.heads = [head]
    mock_driver = isolated_machine.driver
    mock_driver.get_pwm_params = MagicMock(return_value=None)

    assert isolated_machine.get_pwm_settings(head) is None


def test_laser_type_default():
    laser = Laser()
    assert laser.laser_type == LaserType.DIODE


def test_laser_type_supports_pwm():
    assert not LaserType.DIODE.supports_pwm
    assert LaserType.CO2.supports_pwm
    assert LaserType.FIBER.supports_pwm


def test_set_laser_type():
    laser = Laser()
    signal_calls = []

    def handler(sender):
        signal_calls.append(sender)

    laser.changed.connect(handler)
    laser.set_laser_type(LaserType.CO2)
    assert laser.laser_type == LaserType.CO2
    assert len(signal_calls) == 1


def test_set_laser_type_no_signal_on_same():
    laser = Laser()
    signal_calls = []

    def handler(sender):
        signal_calls.append(sender)

    laser.changed.connect(handler)
    laser.set_laser_type(LaserType.DIODE)
    assert len(signal_calls) == 0


def test_laser_type_serialization_roundtrip():
    laser = Laser()
    laser.set_laser_type(LaserType.CO2)
    data = laser.to_dict()
    assert data["laser_type"] == "co2"

    restored = Laser.from_dict(data)
    assert restored.laser_type == LaserType.CO2


def test_laser_type_missing_defaults_to_diode():
    data = {
        "uid": "test-uid",
        "name": "Test Laser",
        "tool_number": 0,
        "max_power": 1000,
    }
    laser = Laser.from_dict(data)
    assert laser.laser_type == LaserType.DIODE


# --- Wavelength / physical power model -------------------------------


def test_laser_type_default_wavelengths():
    """Each LaserType has a published default wavelength."""
    assert LaserType.DIODE.default_wavelength_nm == 455.0
    assert LaserType.CO2.default_wavelength_nm == 10600.0
    assert LaserType.FIBER.default_wavelength_nm == 1064.0


def test_wavelength_defaults_to_zero():
    """A fresh LaserHead has wavelength_nm = 0 (sentinel for the
    laser-type default)."""
    assert LaserHead().wavelength_nm == 0.0


def test_max_power_watts_defaults_to_zero():
    """A fresh LaserHead has max_power_watts = 0 (sentinel for the
    nominal 40 W default)."""
    assert LaserHead().max_power_watts == 0.0


def test_effective_wavelength_uses_explicit_value():
    laser = LaserHead()
    laser.set_laser_type(LaserType.DIODE)
    laser.wavelength_nm = 405.0
    assert laser.effective_wavelength_nm() == 405.0


def test_effective_wavelength_falls_back_to_laser_type():
    laser = LaserHead()
    laser.set_laser_type(LaserType.CO2)
    laser.wavelength_nm = 0.0
    assert laser.effective_wavelength_nm() == 10600.0


def test_effective_max_power_watts_uses_explicit_value():
    laser = LaserHead()
    laser.max_power_watts = 20.0
    assert laser.effective_max_power_watts() == 20.0


def test_effective_max_power_watts_falls_back_to_default():
    laser = LaserHead()
    laser.max_power_watts = 0.0
    assert laser.effective_max_power_watts() == DEFAULT_MAX_POWER_WATTS


def test_watts_at_scales_with_power_fraction():
    laser = LaserHead()
    laser.max_power_watts = 60.0
    assert laser.watts_at(1.0) == 60.0
    assert laser.watts_at(0.5) == 30.0
    assert laser.watts_at(0.0) == 0.0


def test_watts_at_uses_default_when_unconfigured():
    laser = LaserHead()
    laser.max_power_watts = 0.0
    assert laser.watts_at(1.0) == DEFAULT_MAX_POWER_WATTS


def test_wavelength_and_power_serialization_roundtrip():
    laser = LaserHead()
    laser.wavelength_nm = 1064.0
    laser.max_power_watts = 100.0

    data = laser.to_dict()
    assert data["wavelength_nm"] == 1064.0
    assert data["max_power_watts"] == 100.0

    restored = Laser.from_dict(data)
    assert restored.wavelength_nm == 1064.0
    assert restored.max_power_watts == 100.0


def test_wavelength_and_power_missing_fields_use_defaults():
    data = {
        "uid": "test-uid",
        "name": "Test Laser",
        "tool_number": 0,
        "max_power": 1000,
    }
    laser = Laser.from_dict(data)
    assert laser.wavelength_nm == 0.0
    assert laser.max_power_watts == 0.0
    assert (
        laser.effective_wavelength_nm()
        == LaserType.DIODE.default_wavelength_nm
    )
    assert laser.effective_max_power_watts() == DEFAULT_MAX_POWER_WATTS
