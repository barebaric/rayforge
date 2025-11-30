import pytest
from typing import cast
from unittest.mock import Mock
from rayforge.core.varset.var import Var, ValidationError


class TestVar:
    def test_creation_basic(self):
        """Test basic Var creation with a default value."""
        v = Var(key="test", label="Test", var_type=str, default="hello")
        assert v.key == "test"
        assert v.label == "Test"
        assert v.var_type is str
        assert v.default == "hello"
        assert v.value == "hello"

    def test_creation_with_value(self):
        """Test that an explicit value overrides the default on creation."""
        v = Var(key="test", label="Test", var_type=int, default=10, value=20)
        assert v.value == 20

    def test_creation_no_default(self):
        """Test Var creation without a default value."""
        v = Var(key="test", label="Test", var_type=int)
        assert v.default is None
        assert v.value is None

    def test_set_value(self):
        """Test setting a value after creation."""
        v = Var(key="test", label="Test", var_type=int, default=10)
        v.value = 50
        assert v.value == 50
        v.value = 100
        assert v.value == 100

    def test_type_coercion_numeric(self):
        """Test automatic type coercion for numeric types from strings."""
        v_int = Var(key="test_i", label="Test I", var_type=int)
        v_int.value = cast(int, "123")
        assert v_int.value == 123
        assert isinstance(v_int.value, int)

        v_int.value = cast(int, "123.9")
        assert v_int.value == 123
        assert isinstance(v_int.value, int)

        v_float = Var(key="test_f", label="Test F", var_type=float)
        v_float.value = cast(float, "123.45")
        assert v_float.value == 123.45
        assert isinstance(v_float.value, float)

    @pytest.mark.parametrize("s_true", ["true", "1", "on", "yes", "TRUE"])
    def test_type_coercion_boolean_true_str(self, s_true):
        v_bool = Var(key="test_b", label="Test B", var_type=bool)
        v_bool.value = cast(bool, s_true)
        assert v_bool.value is True

    @pytest.mark.parametrize("s_false", ["false", "0", "off", "no", "FALSE"])
    def test_type_coercion_boolean_false_str(self, s_false):
        v_bool = Var(key="test_b", label="Test B", var_type=bool)
        v_bool.value = cast(bool, s_false)
        assert v_bool.value is False

    def test_type_coercion_boolean_numeric(self):
        v_bool = Var(key="test_b", label="Test B", var_type=bool)
        v_bool.value = cast(bool, 1)
        assert v_bool.value is True
        v_bool.value = cast(bool, 0)
        assert v_bool.value is False
        v_bool.value = cast(bool, 5)  # any non-zero number is True
        assert v_bool.value is True

    def test_type_mismatch_error(self):
        """Test that a TypeError is raised for invalid coercions."""
        v_int = Var(key="test", label="Test", var_type=int)
        with pytest.raises(TypeError, match="cannot be coerced"):
            v_int.value = cast(int, "not a number")

        v_bool = Var(key="test_b", label="Test B", var_type=bool)
        with pytest.raises(TypeError, match="cannot be coerced to type bool"):
            v_bool.value = cast(bool, "maybe")

    def test_validation_logic(self):
        """Tests that validation is an explicit step, separate from setting."""

        def range_check(val):
            if val is None or not (0 <= val <= 100):
                raise ValidationError("Value out of range 0-100")

        v = Var(
            key="test",
            label="Test",
            var_type=int,
            validator=range_check,
            value=50,
        )

        v.value = 75
        assert v.value == 75
        v.validate()  # Should not raise

        v.value = 101
        assert v.value == 101, "Value should be updated even if invalid"

        with pytest.raises(ValidationError, match="Value out of range 0-100"):
            v.validate()

        assert v.value == 101

    def test_value_changed_signal(self):
        """Test that the value_changed signal is emitted correctly."""
        v = Var(key="test", label="Test", var_type=int, value=10)
        listener = Mock()
        Var.value_changed.connect(listener, sender=v)

        # 1. Change the value, expect signal
        v.value = 20
        listener.assert_called_once_with(v, new_value=20, old_value=10)

        # 2. Set to same value, expect no signal
        listener.reset_mock()
        v.value = 20
        listener.assert_not_called()

        # 3. Change again
        v.value = 30
        listener.assert_called_once_with(v, new_value=30, old_value=20)

        Var.value_changed.disconnect(listener, sender=v)

    def test_to_dict(self):
        """Test the to_dict method for serializing the definition."""
        v = Var(
            key="test",
            label="Test",
            var_type=str,
            description="A test var",
            default="abc",
            value="xyz",
        )

        # Test without value (definition only)
        data = v.to_dict(include_value=False)
        assert data == {
            "class": "Var",
            "key": "test",
            "label": "Test",
            "description": "A test var",
            "default": "abc",
        }
        assert "value" not in data

        # Test with value
        data_with_val = v.to_dict(include_value=True)
        assert data_with_val["value"] == "xyz"

    def test_repr(self):
        """Test the __repr__ for completeness."""
        v = Var(key="test", label="Test", var_type=int, value=20)
        representation = repr(v)
        assert "key='test'" in representation
        assert "value=20" in representation
        assert "type=int" in representation
