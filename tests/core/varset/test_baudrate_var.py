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

    def test_serialization_round_trip(self):
        """Test serializing and deserializing a BaudrateVar."""
        original_var = BaudrateVar(
            key="baud",
            label="Connection Speed",
            description="The rate of data transfer.",
            default=115200,
        )

        serialized_data = original_var.to_dict()
        assert serialized_data == {
            "class": "BaudrateVar",
            "key": "baud",
            "label": "Connection Speed",
            "description": "The rate of data transfer.",
            "default": 115200,
            "min_val": 300,
            "max_val": 4000000,
        }

        rehydrated_var = VarSet._create_var_from_dict(serialized_data)

        assert isinstance(rehydrated_var, BaudrateVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.description == original_var.description
        assert rehydrated_var.default == original_var.default
