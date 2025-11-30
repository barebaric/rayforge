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

    def test_serialization_round_trip(self):
        """Test serializing and deserializing a TextAreaVar."""
        original_var = TextAreaVar(
            key="gcode",
            label="G-Code Script",
            description="A multi-line script.",
            default="G0 X10 Y10",
        )

        serialized_data = original_var.to_dict()
        assert serialized_data == {
            "class": "TextAreaVar",
            "key": "gcode",
            "label": "G-Code Script",
            "description": "A multi-line script.",
            "default": "G0 X10 Y10",
        }

        rehydrated_var = VarSet._create_var_from_dict(serialized_data)

        assert isinstance(rehydrated_var, TextAreaVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.description == original_var.description
        assert rehydrated_var.default == original_var.default
