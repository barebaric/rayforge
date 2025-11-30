import pytest
from rayforge.core.varset.floatvar import FloatVar, SliderFloatVar
from rayforge.core.varset.var import ValidationError
from rayforge.core.varset.varset import VarSet


class TestFloatVar:
    def test_validation(self):
        """Test FloatVar with min/max bounds and an extra validator."""
        v = FloatVar(
            key="test_float", label="Test", min_val=10.5, max_val=20.5
        )
        v.value = 15.0
        v.validate()

        v.value = 10.4
        with pytest.raises(ValidationError, match="at least 10.5"):
            v.validate()

    def test_serialization_and_rehydration(self):
        """Test serializing (with and without value) and deserializing."""
        original_var = FloatVar(
            key="speed",
            label="Feed Rate",
            description="Speed in mm/min.",
            default=1000.0,
            min_val=0.1,
            max_val=5000.0,
        )
        original_var.value = 2500.5

        # Test serialization of definition (default behavior)
        serialized_def = original_var.to_dict()
        assert "value" not in serialized_def
        assert serialized_def == {
            "class": "FloatVar",
            "key": "speed",
            "label": "Feed Rate",
            "description": "Speed in mm/min.",
            "default": 1000.0,
            "min_val": 0.1,
            "max_val": 5000.0,
        }

        # Test serialization of state (include_value=True)
        serialized_state = original_var.to_dict(include_value=True)
        assert serialized_state["value"] == 2500.5

        # Test rehydration from definition
        rehydrated_var = VarSet._create_var_from_dict(serialized_def)
        assert isinstance(rehydrated_var, FloatVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.default == original_var.default
        assert rehydrated_var.min_val == original_var.min_val
        assert rehydrated_var.max_val == original_var.max_val
        assert rehydrated_var.value == original_var.default


class TestSliderFloatVar:
    def test_validation(self):
        """Test SliderFloatVar to ensure it behaves like a FloatVar."""
        v = SliderFloatVar(
            key="slider", label="Slider", min_val=0.0, max_val=1.0
        )
        v.value = 0.5
        v.validate()

        v.value = 1.1
        with pytest.raises(ValidationError, match="at most 1.0"):
            v.validate()

    def test_serialization_and_rehydration(self):
        """Test serializing (with and without value) and deserializing."""
        original_var = SliderFloatVar(
            key="opacity",
            label="Opacity",
            description="From 0 to 1.",
            default=0.8,
            min_val=0.0,
            max_val=1.0,
        )
        original_var.value = 0.5

        # Test serialization of definition (default behavior)
        serialized_def = original_var.to_dict()
        assert "value" not in serialized_def
        assert serialized_def["class"] == "SliderFloatVar"

        # Test serialization of state (include_value=True)
        serialized_state = original_var.to_dict(include_value=True)
        assert serialized_state["value"] == 0.5

        # Test rehydration from definition
        rehydrated_var = VarSet._create_var_from_dict(serialized_def)
        assert isinstance(rehydrated_var, SliderFloatVar)
        assert rehydrated_var.max_val == 1.0
        assert rehydrated_var.value == 0.8
