from rayforge.core.varset.boolvar import BoolVar
from rayforge.core.varset.varset import VarSet


class TestBoolVar:
    def test_creation(self):
        """Test BoolVar creation and default value handling."""
        v = BoolVar(key="test", label="Test", default=True)
        assert v.value is True

        v.value = False
        assert v.value is False

    def test_serialization_round_trip(self):
        """Test serializing and deserializing a BoolVar."""
        original_var = BoolVar(
            key="enabled",
            label="Is Enabled",
            description="A boolean flag.",
            default=False,
        )

        serialized_data = original_var.to_dict()
        assert serialized_data == {
            "class": "BoolVar",
            "key": "enabled",
            "label": "Is Enabled",
            "description": "A boolean flag.",
            "default": False,
        }

        rehydrated_var = VarSet._create_var_from_dict(serialized_data)

        assert isinstance(rehydrated_var, BoolVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.description == original_var.description
        assert rehydrated_var.default == original_var.default
