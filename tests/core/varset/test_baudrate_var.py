import pytest
from rayforge.core.varset.baudratevar import BaudrateVar, ValidationError
from rayforge.core.varset.varset import VarSet


class TestBaudrateVar:
    def test_validation_default_choices(self):
        """Test BaudrateVar validation with default standard rates."""
        v = BaudrateVar(key="baud", label="Baud Rate")

        v.value = 115200
        v.validate()

        v.value = 9601
        with pytest.raises(ValidationError, match="not a standard baud rate"):
            v.validate()

    def test_validation_custom_choices(self):
        """Test BaudrateVar with custom choices list."""
        v = BaudrateVar(key="baud", label="Baud Rate", choices=[9600, 115200])

        v.value = 9600
        v.validate()

        v.value = 19200
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
            "choices": [
                9600,
                19200,
                38400,
                57600,
                115200,
                230400,
                460800,
                921600,
                1000000,
                1843200,
            ],
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
        assert rehydrated_var.value == original_var.default

    def test_rehydration_preserves_custom_choices(self):
        """Test that custom choices survive serialization round-trip."""
        original = BaudrateVar(
            key="baud", label="Baud", choices=[9600, 115200]
        )
        data = original.to_dict()
        rehydrated = VarSet._create_var_from_dict(data)
        assert isinstance(rehydrated, BaudrateVar)
        assert rehydrated.choices == [9600, 115200]
