import pytest
from rayforge.core.varset.serialportvar import SerialPortVar, ValidationError
from rayforge.core.varset.varset import VarSet


class TestSerialPortVar:
    def test_validation(self):
        """Test SerialPortVar validation logic."""
        v = SerialPortVar(key="port", label="Port")
        v.value = "/dev/ttyUSB0"
        v.validate()

        v.value = None
        with pytest.raises(ValidationError, match="cannot be empty"):
            v.validate()

    def test_serialization_and_rehydration(self):
        """Test serializing (with and without value) and deserializing."""
        original_var = SerialPortVar(
            key="device",
            label="Device Port",
            description="The serial device path.",
            default="COM1",
        )
        original_var.value = "/dev/ttyACM0"  # Set a non-default value

        # Test serialization of definition (default behavior)
        serialized_def = original_var.to_dict()
        assert "value" not in serialized_def
        assert serialized_def == {
            "class": "SerialPortVar",
            "key": "device",
            "label": "Device Port",
            "description": "The serial device path.",
            "default": "COM1",
        }

        # Test serialization of state (include_value=True)
        serialized_state = original_var.to_dict(include_value=True)
        assert serialized_state["value"] == "/dev/ttyACM0"

        # Test rehydration from definition
        rehydrated_var = VarSet._create_var_from_dict(serialized_def)
        assert isinstance(rehydrated_var, SerialPortVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.description == original_var.description
        assert rehydrated_var.default == original_var.default
        assert rehydrated_var.value == original_var.default
