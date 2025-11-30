from rayforge.core.varset.textareavar import TextAreaVar
from rayforge.core.varset.varset import VarSet


class TestTextAreaVar:
    def test_creation(self):
        """Test TextAreaVar creation and default value handling."""
        v = TextAreaVar(
            key="test",
            label="Test",
            default="hello\nworld",
        )
        assert v.value == "hello\nworld"

    def test_serialization_and_rehydration(self):
        """Test serializing (with and without value) and deserializing."""
        original_var = TextAreaVar(
            key="gcode",
            label="G-Code Script",
            description="A multi-line script.",
            default="G0 X10 Y10",
        )
        original_var.value = "G1 F1000\nG1 X20"  # Set a non-default value

        # Test serialization of definition (default behavior)
        serialized_def = original_var.to_dict()
        assert "value" not in serialized_def
        assert serialized_def == {
            "class": "TextAreaVar",
            "key": "gcode",
            "label": "G-Code Script",
            "description": "A multi-line script.",
            "default": "G0 X10 Y10",
        }

        # Test serialization of state (include_value=True)
        serialized_state = original_var.to_dict(include_value=True)
        assert serialized_state["value"] == "G1 F1000\nG1 X20"

        # Test rehydration from definition
        rehydrated_var = VarSet._create_var_from_dict(serialized_def)
        assert isinstance(rehydrated_var, TextAreaVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.description == original_var.description
        assert rehydrated_var.default == original_var.default
        assert rehydrated_var.value == original_var.default
