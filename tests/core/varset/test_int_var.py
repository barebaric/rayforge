import pytest
from rayforge.core.varset.intvar import IntVar
from rayforge.core.varset.var import ValidationError
from rayforge.core.varset.varset import VarSet


class TestIntVar:
    def test_validation(self):
        """Test IntVar with min/max bounds."""
        v = IntVar(key="test_int", label="Test", min_val=10, max_val=20)
        v.value = 15
        v.validate()

        v.value = 9
        with pytest.raises(ValidationError, match="at least 10"):
            v.validate()

        v.value = 21
        with pytest.raises(ValidationError, match="at most 20"):
            v.validate()

        v.value = None
        v.validate()  # Should pass, as checks are guarded

    def test_with_non_nullable_validator(self):
        """Test an IntVar with a validator that rejects None."""

        def not_none(v):
            if v is None:
                raise ValidationError("Value cannot be None")

        v = IntVar(key="test", label="Test", validator=not_none)
        v.value = 10
        v.validate()

        v.value = None
        with pytest.raises(ValidationError, match="Value cannot be None"):
            v.validate()

    def test_serialization_and_rehydration(self):
        """Test serializing (with and without value) and deserializing."""
        original_var = IntVar(
            key="retries",
            label="Retry Count",
            description="Number of times to retry.",
            default=3,
            min_val=0,
            max_val=10,
        )
        original_var.value = 7  # Set a non-default value

        # Test serialization of definition (default behavior)
        serialized_def = original_var.to_dict()
        assert "value" not in serialized_def
        assert serialized_def == {
            "class": "IntVar",
            "key": "retries",
            "label": "Retry Count",
            "description": "Number of times to retry.",
            "default": 3,
            "min_val": 0,
            "max_val": 10,
        }

        # Test serialization of state (include_value=True)
        serialized_state = original_var.to_dict(include_value=True)
        assert serialized_state["value"] == 7

        # Test rehydration from definition
        rehydrated_var = VarSet._create_var_from_dict(serialized_def)
        assert isinstance(rehydrated_var, IntVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.default == original_var.default
        assert rehydrated_var.min_val == original_var.min_val
        assert rehydrated_var.max_val == original_var.max_val
        assert rehydrated_var.value == original_var.default
