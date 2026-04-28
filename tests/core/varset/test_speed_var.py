import pytest
from rayforge.core.varset.speedvar import SpeedVar
from rayforge.core.varset.var import ValidationError
from rayforge.core.varset.varset import VarSet


class TestSpeedVar:
    def test_validation(self):
        """Test SpeedVar with min/max bounds."""
        v = SpeedVar(
            key="cut_speed",
            label="Cut Speed",
            min_val=1,
            max_val=1000,
        )
        v.value = 500
        v.validate()

        v.value = 0
        with pytest.raises(ValidationError, match="at least 1"):
            v.validate()

        v.value = 1001
        with pytest.raises(ValidationError, match="at most 1000"):
            v.validate()

        v.value = None
        v.validate()

    def test_serialization_and_rehydration(self):
        """Test serializing (with and without value) and deserializing."""
        original_var = SpeedVar(
            key="cut_speed",
            label="Cut Speed",
            description="Speed of the cutting head.",
            default=100,
            min_val=1,
            max_val=5000,
        )
        original_var.value = 250

        serialized_def = original_var.to_dict()
        assert "value" not in serialized_def
        assert serialized_def == {
            "class": "SpeedVar",
            "key": "cut_speed",
            "label": "Cut Speed",
            "description": "Speed of the cutting head.",
            "default": 100,
            "min_val": 1,
            "max_val": 5000,
        }

        serialized_state = original_var.to_dict(include_value=True)
        assert serialized_state["value"] == 250

        rehydrated_var = VarSet._create_var_from_dict(serialized_def)
        assert isinstance(rehydrated_var, SpeedVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.description == original_var.description
        assert rehydrated_var.default == original_var.default
        assert rehydrated_var.min_val == original_var.min_val
        assert rehydrated_var.max_val == original_var.max_val
        assert rehydrated_var.value == original_var.default
