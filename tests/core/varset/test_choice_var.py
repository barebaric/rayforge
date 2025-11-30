import pytest
from rayforge.core.varset.choicevar import ChoiceVar
from rayforge.core.varset.var import ValidationError
from rayforge.core.varset.varset import VarSet


class TestChoiceVar:
    def test_validation(self):
        """Test ChoiceVar validation."""
        choices = ["A", "B", "C"]
        v = ChoiceVar(key="choice", label="Choice", choices=choices, value="A")

        v.validate()  # Should pass

        v.value = "B"
        v.validate()  # Should pass

        v.value = "D"
        with pytest.raises(ValidationError, match="'D' is not a valid choice"):
            v.validate()

        v.value = None
        v.validate()  # None should be allowed by default

    def test_serialization_and_rehydration(self):
        """Test serializing (with and without value) and deserializing."""
        original_var = ChoiceVar(
            key="mode",
            label="Operating Mode",
            description="Select a mode.",
            choices=["Fast", "Slow", "Balanced"],
            default="Balanced",
        )
        original_var.value = "Fast"  # Set a non-default value

        # Test serialization of definition (default behavior)
        serialized_def = original_var.to_dict()
        assert "value" not in serialized_def
        assert serialized_def == {
            "class": "ChoiceVar",
            "key": "mode",
            "label": "Operating Mode",
            "description": "Select a mode.",
            "default": "Balanced",
            "choices": ["Fast", "Slow", "Balanced"],
        }

        # Test serialization of state (include_value=True)
        serialized_state = original_var.to_dict(include_value=True)
        assert serialized_state["value"] == "Fast"

        # Test rehydration from definition
        rehydrated_var = VarSet._create_var_from_dict(serialized_def)
        assert isinstance(rehydrated_var, ChoiceVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.description == original_var.description
        assert rehydrated_var.default == original_var.default
        assert rehydrated_var.choices == original_var.choices
        assert rehydrated_var.value == original_var.default
