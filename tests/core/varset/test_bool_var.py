from rayforge.core.varset.boolvar import BoolVar
from rayforge.core.varset.varset import VarSet


class TestBoolVar:
    def test_creation(self):
        """Test BoolVar creation and default value handling."""
        v = BoolVar(key="test", label="Test", default=True)
        assert v.value is True

        v.value = False
        assert v.value is False

    def test_serialization_and_rehydration(self):
        """Test serializing (with and without value) and deserializing."""
        original_var = BoolVar(
            key="enabled",
            label="Is Enabled",
            description="A boolean flag.",
            default=False,
        )
        original_var.value = True  # Set a non-default value

        # Test serialization of definition (default behavior)
        serialized_def = original_var.to_dict()
        assert "value" not in serialized_def
        assert serialized_def == {
            "class": "BoolVar",
            "key": "enabled",
            "label": "Is Enabled",
            "description": "A boolean flag.",
            "default": False,
        }

        # Test serialization of state (include_value=True)
        serialized_state = original_var.to_dict(include_value=True)
        assert serialized_state["value"] is True

        # Test rehydration from definition
        rehydrated_var = VarSet._create_var_from_dict(serialized_def)
        assert isinstance(rehydrated_var, BoolVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.description == original_var.description
        assert rehydrated_var.default == original_var.default
        assert rehydrated_var.value == original_var.default
