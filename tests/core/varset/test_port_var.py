import pytest
from rayforge.core.varset.portvar import PortVar, ValidationError
from rayforge.core.varset.varset import VarSet


class TestPortVar:
    def test_validation(self):
        """Test PortVar for network ports, checking its built-in validation."""
        v = PortVar(key="port", label="Network Port")
        v.value = 8080
        v.validate()

        v.value = 0
        with pytest.raises(ValidationError, match="at least 1"):
            v.validate()

        v.value = 65536
        with pytest.raises(ValidationError, match="at most 65535"):
            v.validate()

    def test_serialization_and_rehydration(self):
        """Test serializing (with and without value) and deserializing."""
        original_var = PortVar(
            key="http_port",
            label="HTTP Port",
            description="The web server port.",
            default=80,
        )
        original_var.value = 8080  # Set a non-default value

        # Test serialization of definition (default behavior)
        serialized_def = original_var.to_dict()
        assert "value" not in serialized_def
        # PortVar inherits from IntVar, so it will serialize the bounds
        assert serialized_def == {
            "class": "PortVar",
            "key": "http_port",
            "label": "HTTP Port",
            "description": "The web server port.",
            "default": 80,
            "min_val": 1,
            "max_val": 65535,
        }

        # Test serialization of state (include_value=True)
        serialized_state = original_var.to_dict(include_value=True)
        assert serialized_state["value"] == 8080

        # Test rehydration from definition
        rehydrated_var = VarSet._create_var_from_dict(serialized_def)
        assert isinstance(rehydrated_var, PortVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.description == original_var.description
        assert rehydrated_var.default == original_var.default
        assert rehydrated_var.value == original_var.default
