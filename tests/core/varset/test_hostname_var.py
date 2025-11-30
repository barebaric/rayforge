import pytest
from unittest.mock import patch
from rayforge.core.varset.hostnamevar import HostnameVar, ValidationError
from rayforge.core.varset.varset import VarSet


class TestHostnameVar:
    @patch("rayforge.core.varset.hostnamevar.is_valid_hostname_or_ip")
    def test_validation(self, mock_is_valid):
        """Test HostnameVar validation logic."""
        mock_is_valid.side_effect = lambda h: h in ["valid.com", "1.2.3.4"]
        v = HostnameVar(key="host", label="Host")

        v.value = "valid.com"
        v.validate()

        v.value = "invalid-hostname"
        with pytest.raises(
            ValidationError, match="Invalid hostname or IP address format"
        ):
            v.validate()

        v.value = None
        with pytest.raises(ValidationError, match="cannot be empty"):
            v.validate()

    def test_serialization_and_rehydration(self):
        """Test serializing (with and without value) and deserializing."""
        original_var = HostnameVar(
            key="server_ip",
            label="Server Address",
            description="Hostname or IP.",
            default="localhost",
        )
        original_var.value = "127.0.0.1"  # Set a non-default value

        # Test serialization of definition (default behavior)
        serialized_def = original_var.to_dict()
        assert "value" not in serialized_def
        assert serialized_def == {
            "class": "HostnameVar",
            "key": "server_ip",
            "label": "Server Address",
            "description": "Hostname or IP.",
            "default": "localhost",
        }

        # Test serialization of state (include_value=True)
        serialized_state = original_var.to_dict(include_value=True)
        assert serialized_state["value"] == "127.0.0.1"

        # Test rehydration from definition
        rehydrated_var = VarSet._create_var_from_dict(serialized_def)
        assert isinstance(rehydrated_var, HostnameVar)
        assert rehydrated_var.key == original_var.key
        assert rehydrated_var.label == original_var.label
        assert rehydrated_var.description == original_var.description
        assert rehydrated_var.default == original_var.default
        assert rehydrated_var.value == original_var.default
