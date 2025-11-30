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

    def test_serialization_round_trip(self):
        """Test serializing and deserializing a SerialPortVar."""
        original_var = SerialPortVar(
            key="device",
            label="Device Port",
            description="The serial device path.",
            default="COM1",
        )

        serialized_data = original_var.to_dict()
        assert serialized_data == {
            "class": "SerialPortVar",
            "key": "device",
            "label": "Device Port",
            "description": "The serial device path.",
            "default": "COM1",
        }

        rehydrated_var = VarSet._create_var_from_dict(serialized_data)

        assert isinstance(rehydrated_var, SerialPortVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.description == original_var.description
        assert rehydrated_var.default == original_var.default
