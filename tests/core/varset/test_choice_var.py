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

    def test_serialization_round_trip(self):
        """Test serializing and deserializing a ChoiceVar."""
        original_var = ChoiceVar(
            key="mode",
            label="Operating Mode",
            description="Select a mode.",
            choices=["Fast", "Slow", "Balanced"],
            default="Balanced",
        )

        serialized_data = original_var.to_dict()
        assert serialized_data == {
            "class": "ChoiceVar",
            "key": "mode",
            "label": "Operating Mode",
            "description": "Select a mode.",
            "default": "Balanced",
            "choices": ["Fast", "Slow", "Balanced"],
        }

        rehydrated_var = VarSet._create_var_from_dict(serialized_data)

        assert isinstance(rehydrated_var, ChoiceVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.description == original_var.description
        assert rehydrated_var.default == original_var.default
        assert rehydrated_var.choices == original_var.choices
