from rayforge.machine.models.laser import Laser


class TestLaser:
    """Test suite for Laser model."""

    def test_laser_initialization(self):
        """Test that a new laser initializes with default values."""
        laser = Laser()

        assert laser.name == "Laser Head"
        assert laser.tool_number == 0
        assert laser.max_power == 1000
        assert laser.frame_power_percent == 0
        assert laser.focus_power_percent == 0
        assert laser.spot_size_mm == (0.1, 0.1)
        assert laser.uid is not None

    def test_set_focus_power(self):
        """
        Test that set_focus_power updates the focus power percent and sends
        signal.
        """
        laser = Laser()

        # Mock the changed signal
        signal_calls = []

        def signal_handler(sender):
            signal_calls.append(sender)

        laser.changed.connect(signal_handler)

        # Set focus power
        laser.set_focus_power(0.05)  # 5% as a decimal

        assert laser.focus_power_percent == 0.05
        assert len(signal_calls) == 1
        assert signal_calls[0] is laser

    def test_set_focus_power_zero(self):
        """Test that setting focus power to 0 works correctly."""
        laser = Laser()

        laser.set_focus_power(0.0)

        assert laser.focus_power_percent == 0.0

    def test_set_focus_power_multiple_values(self):
        """Test setting focus power to various values."""
        laser = Laser()

        test_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

        for value in test_values:
            laser.set_focus_power(value)
            assert laser.focus_power_percent == value

    def test_laser_serialization_includes_focus_power(self):
        """Test that focus power is included in serialization."""
        laser = Laser()
        laser.set_focus_power(0.42)  # 42% as a decimal

        data = laser.to_dict()

        # Check that only percentage fields are in serialization
        assert "focus_power_percent" in data
        assert data["focus_power_percent"] == 42  # 0.42 * 100 = 42%
        # Old gcode fields should not be in serialization
        assert "focus_power" not in data

    def test_laser_deserialization_focus_power(self):
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

        # Check percentage values (converted from old gcode format)
        assert laser.focus_power_percent == 35 / 1500  # 35/1500 = 0.0233
        assert laser.frame_power_percent == 20 / 1500  # 20/1500 = 0.0133

        assert laser.uid == "test-uid"
        assert laser.name == "Test Laser"
        assert laser.tool_number == 1
        assert laser.max_power == 1500
        assert laser.spot_size_mm == [0.15, 0.15]

    def test_laser_deserialization_default_focus_power(self):
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

    def test_laser_roundtrip_serialization(self):
        """Test that focus power survives a full serialization roundtrip."""
        original_laser = Laser()
        original_laser.set_focus_power(0.67)  # 67% as a decimal
        original_laser.name = "Roundtrip Test"

        # Serialize to dict
        data = original_laser.to_dict()

        # Deserialize back to laser
        new_laser = Laser.from_dict(data)

        assert (
            new_laser.focus_power_percent == original_laser.focus_power_percent
        )
        assert new_laser.name == original_laser.name
        assert new_laser.uid == original_laser.uid

    def test_backward_compatibility_old_format(self):
        """Test that old format data (without _percent fields) is converted
        correctly.
        """
        # Old format data with gcode units
        old_data = {
            "uid": "old-uid",
            "name": "Old Laser",
            "tool_number": 1,
            "max_power": 1000,
            "frame_power": 100,  # 100 in gcode units
            "focus_power": 500,  # 500 in gcode units
            "spot_size_mm": [0.2, 0.2],
        }

        laser = Laser.from_dict(old_data)

        assert laser.frame_power_percent == 0.1  # 100/1000 = 0.1
        assert laser.focus_power_percent == 0.5  # 500/1000 = 0.5

    def test_new_format_deserialization(self):
        """Test that new format data (with _percent fields) is deserialized
        correctly.
        """
        # New format data with percentage values
        new_data = {
            "uid": "new-uid",
            "name": "New Laser",
            "tool_number": 2,
            "max_power": 2000,
            "frame_power": 400,  # Should be ignored in favor of _percent
            "frame_power_percent": 0.2,
            "focus_power": 1000,  # Should be ignored in favor of _percent
            "focus_power_percent": 0.5,
            "spot_size_mm": [0.3, 0.3],
        }

        laser = Laser.from_dict(new_data)

        assert laser.frame_power_percent == 0.002  # 0.2 / 100 = 0.002
        assert laser.focus_power_percent == 0.005  # 0.5 / 100 = 0.005

    def test_serialization_includes_percent_fields(self):
        """Test that serialization includes only percentage fields."""
        laser = Laser()
        laser.set_frame_power(0.3)  # 30% as decimal
        laser.set_focus_power(0.75)  # 75% as decimal

        data = laser.to_dict()

        # Only percentage fields should be in serialization
        assert "frame_power_percent" in data
        assert "focus_power_percent" in data
        assert data["frame_power_percent"] == 30  # 0.3 * 100 = 30%
        assert data["focus_power_percent"] == 75  # 0.75 * 100 = 75%

        # Old gcode fields should not be in serialization
        assert "frame_power" not in data
        assert "focus_power" not in data

    def test_laser_forward_compatibility_with_extra_fields(self):
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

        # Verify extra fields are stored
        assert laser.extra["future_field_string"] == "some value"
        assert laser.extra["future_field_number"] == 42
        assert laser.extra["future_field_dict"] == {"nested": "data"}

        # Verify extra fields are re-serialized
        data = laser.to_dict()
        assert data["future_field_string"] == "some value"
        assert data["future_field_number"] == 42
        assert data["future_field_dict"] == {"nested": "data"}

    def test_laser_backward_compat_missing_optional_fields(self):
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

        # Verify defaults are applied for missing optional fields
        assert laser.name == "Old Laser"
        assert laser.frame_power_percent == 0
        assert laser.focus_power_percent == 0
        assert laser.spot_size_mm == (0.1, 0.1)
        assert laser.extra == {}
