import pytest
from unittest.mock import patch
from rayforge.core.varset.baudratevar import BaudrateVar, ValidationError
from rayforge.core.varset.varset import VarSet


class TestBaudrateVar:
    @patch("rayforge.machine.transport.serial.SerialTransport.list_baud_rates")
    def test_validation(self, mock_list_rates):
        """Test BaudrateVar against a mocked list of standard rates."""
        mock_list_rates.return_value = [9600, 19200, 115200]
        v = BaudrateVar(key="baud", label="Baud Rate")

        v.value = 115200
        v.validate()

        v.value = 9601
        with pytest.raises(ValidationError, match="not a standard baud rate"):
            v.validate()

    def test_serialization_and_rehydration(self):
        """Test serializing (with and without value) and deserializing."""
        original_var = BaudrateVar(
            key="baud",
            label="Connection Speed",
            description="The rate of data transfer.",
            default=115200,
        )
        original_var.value = 9600  # Set a non-default value

        # Test serialization of definition (default behavior)
        serialized_def = original_var.to_dict()
        assert "value" not in serialized_def
        assert serialized_def == {
            "class": "BaudrateVar",
            "key": "baud",
            "label": "Connection Speed",
            "description": "The rate of data transfer.",
            "default": 115200,
            "min_val": 300,
            "max_val": 4000000,
        }

        # Test serialization of state (include_value=True)
        serialized_state = original_var.to_dict(include_value=True)
        assert serialized_state["value"] == 9600

        # Test rehydration from definition
        rehydrated_var = VarSet._create_var_from_dict(serialized_def)
        assert isinstance(rehydrated_var, BaudrateVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.description == original_var.description
        assert rehydrated_var.default == original_var.default
        assert (
            rehydrated_var.value == original_var.default
        )  # Rehydrated value is the default
