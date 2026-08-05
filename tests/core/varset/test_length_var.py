import pytest

from rayforge.core.varset.floatvar import FloatVar
from rayforge.core.varset.lengthvar import LengthVar
from rayforge.core.varset.var import ValidationError
from rayforge.core.varset.varset import VarSet


class TestLengthVar:
    def test_is_float_var(self):
        var = LengthVar(key="offset_mm", label="Offset")
        assert isinstance(var, FloatVar)

    def test_validation(self):
        """Test LengthVar with min/max bounds."""
        v = LengthVar(
            key="offset_mm",
            label="Offset",
            min_val=0.0,
            max_val=100.0,
        )
        v.value = 5.5
        v.validate()

        v.value = -1.0
        with pytest.raises(ValidationError, match="at least 0"):
            v.validate()

        v.value = 101.0
        with pytest.raises(ValidationError, match="at most 100"):
            v.validate()

        v.value = None
        v.validate()

    def test_serialization_and_rehydration(self):
        """Test serializing (with and without value) and deserializing."""
        original_var = LengthVar(
            key="offset_mm",
            label="Offset",
            description="Shifts the cut path.",
            default=0.0,
            min_val=-10.0,
            max_val=10.0,
        )
        original_var.value = 2.5

        serialized_def = original_var.to_dict()
        assert "value" not in serialized_def
        assert serialized_def == {
            "class": "LengthVar",
            "key": "offset_mm",
            "label": "Offset",
            "description": "Shifts the cut path.",
            "default": 0.0,
            "min_val": -10.0,
            "max_val": 10.0,
        }

        serialized_state = original_var.to_dict(include_value=True)
        assert serialized_state["value"] == 2.5

        rehydrated_var = VarSet._create_var_from_dict(serialized_def)
        assert isinstance(rehydrated_var, LengthVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.description == original_var.description
        assert rehydrated_var.default == original_var.default
        assert rehydrated_var.min_val == original_var.min_val
        assert rehydrated_var.max_val == original_var.max_val
        assert rehydrated_var.value == original_var.default
