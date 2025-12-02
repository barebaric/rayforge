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
        assert v.value == "hello"  # Effective value is default
        assert v.raw_value is None  # No explicit value set

    def test_creation_with_value(self):
        """Test that an explicit value overrides the default on creation."""
        v = Var(key="test", label="Test", var_type=int, default=10, value=20)
        assert v.default == 10
        assert v.raw_value == 20
        assert v.value == 20  # Effective value is the explicit one

    def test_creation_no_default(self):
        """Test Var creation without a default value."""
        v = Var(key="test", label="Test", var_type=int)
        assert v.default is None
        assert v.raw_value is None
        assert v.value is None

    def test_set_value(self):
        """Test setting an explicit value after creation."""
        v = Var(key="test", label="Test", var_type=int, default=10)
        assert v.value == 10
        v.value = 50
        assert v.raw_value == 50
        assert v.value == 50
        v.value = 100
        assert v.raw_value == 100
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

    def test_value_changed_signal_on_explicit_set(self):
        """Test that the value_changed signal is emitted correctly."""
        v = Var(key="test", label="Test", var_type=int, value=10)
        listener = Mock()
        v.value_changed.connect(listener)

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

        v.value_changed.disconnect(listener)

    def test_definition_changed_signal(self):
        """Test the new definition_changed signal for key, label, and desc."""
        v = Var(
            key="old_key",
            label="Old Label",
            var_type=str,
            description="Old Desc",
        )
        listener = Mock()
        v.definition_changed.connect(listener)

        # Change key
        v.key = "new_key"
        listener.assert_called_once_with(v, property="key")
        listener.reset_mock()

        # Set key to same value (no signal)
        v.key = "new_key"
        listener.assert_not_called()

        # Change label
        v.label = "New Label"
        listener.assert_called_once_with(v, property="label")
        listener.reset_mock()

        # Set label to same value (no signal)
        v.label = "New Label"
        listener.assert_not_called()

        # Change description
        v.description = "New Desc"
        listener.assert_called_once_with(v, property="description")
        listener.reset_mock()

        # Set description to same value (no signal)
        v.description = "New Desc"
        listener.assert_not_called()

        v.definition_changed.disconnect(listener)

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

    # --- NEW TESTS TO VERIFY CONTRACT ---

    def test_value_falls_back_to_default(self):
        """
        Tests that the `value` property correctly returns the default when no
        explicit value has been set.
        """
        v = Var(key="fallback", label="Fallback", var_type=int, default=100)
        assert v.value == 100, "Effective value should be the default"
        assert v.raw_value is None, "No explicit value should be stored"

    def test_changing_default_triggers_both_signals_when_using_default(self):
        """
        Verifies that changing the default emits both signals
        if the Var is currently using the default.
        """
        # 1. Setup: Var is using its default value
        v = Var(
            key="test_default", label="Test Default", var_type=int, default=10
        )
        assert v.value == 10
        assert v.raw_value is None

        value_listener = Mock()
        def_listener = Mock()
        v.value_changed.connect(value_listener)
        v.definition_changed.connect(def_listener)

        # 2. Action: Change the default
        v.default = 20

        # 3. Assertions
        assert v.default == 20, "Default should be updated"
        assert v.value == 20, "Effective value should now be the new default"
        value_listener.assert_called_once_with(v, new_value=20, old_value=10)
        def_listener.assert_called_once_with(v, property="default")

        v.value_changed.disconnect(value_listener)
        v.definition_changed.disconnect(def_listener)

    def test_changing_default_triggers_only_definition_signal_when_overridden(
        self,
    ):
        """
        Verifies that changing the default value ONLY emits definition_changed
        if an explicit value is already set.
        """
        # 1. Setup: Var has an explicit value
        v = Var(
            key="test_default",
            label="Test Default",
            var_type=int,
            default=10,
            value=50,
        )
        assert v.value == 50

        value_listener = Mock()
        def_listener = Mock()
        v.value_changed.connect(value_listener)
        v.definition_changed.connect(def_listener)

        # 2. Action: Change the default
        v.default = 20

        # 3. Assertions
        assert v.default == 20, "Default should be updated"
        assert v.value == 50, "Effective value should remain the explicit one"
        value_listener.assert_not_called()
        def_listener.assert_called_once_with(v, property="default")

        v.value_changed.disconnect(value_listener)
        v.definition_changed.disconnect(def_listener)

    def test_setting_value_to_none_falls_back_to_default_and_signals(self):
        """
        CRITICAL TEST: Verifies that clearing an explicit value by setting it
        to None causes the Var to fall back to default and emit a signal.
        """
        # 1. Setup: Var has an explicit value
        v = Var(
            key="test_fallback",
            label="Test Fallback",
            var_type=int,
            default=10,
            value=50,
        )
        assert v.value == 50

        listener = Mock()
        v.value_changed.connect(listener)

        # 2. Action: Set explicit value to None
        v.value = None

        # 3. Assertions
        assert v.raw_value is None, "Explicit value should now be None"
        assert v.value == 10, "Effective value should fall back to the default"
        listener.assert_called_once_with(v, new_value=10, old_value=50)

        v.value_changed.disconnect(listener)
