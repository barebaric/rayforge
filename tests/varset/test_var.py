import pytest
from typing import cast
from unittest.mock import patch
from rayforge.core.varset.var import Var, ValidationError
from rayforge.core.varset.intvar import IntVar
from rayforge.core.varset.floatvar import FloatVar, SliderFloatVar
from rayforge.core.varset.choicevar import ChoiceVar
from rayforge.core.varset.hostnamevar import HostnameVar
from rayforge.core.varset.serialportvar import SerialPortVar
from rayforge.core.varset.portvar import PortVar
from rayforge.core.varset.baudratevar import BaudrateVar


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

    def test_to_dict_and_repr(self):
        """Test the to_dict method and __repr__ for completeness."""
        v = Var(
            key="test",
            label="Test",
            var_type=int,
            description="A test var",
            default=10,
            value=20,
        )
        data = v.to_dict()

        assert data["key"] == "test"
        assert data["label"] == "Test"
        assert data["var_type"] is int
        assert data["description"] == "A test var"
        assert data["default"] == 10
        assert data["value"] == 20
        assert data["validator"] is None

        representation = repr(v)
        assert "key='test'" in representation
        assert "value=20" in representation
        assert "type=int" in representation

    # --- Tests for Var subclasses ---

    def test_int_var(self):
        """Test IntVar with min/max bounds and an extra validator."""
        v = IntVar(key="test_int", label="Test", min_val=10, max_val=20)
        v.value = 15
        v.validate()

        v.value = 9
        with pytest.raises(ValidationError, match="at least 10"):
            v.validate()

        v.value = 21
        with pytest.raises(ValidationError, match="at most 20"):
            v.validate()

        v.value = None
        v.validate()  # Should pass, as checks are guarded

    def test_int_var_with_non_nullable_validator(self):
        """Test an IntVar with a validator that rejects None."""

        def not_none(v):
            if v is None:
                raise ValidationError("Value cannot be None")

        v = IntVar(key="test", label="Test", validator=not_none)
        v.value = 10
        v.validate()

        v.value = None
        with pytest.raises(ValidationError, match="Value cannot be None"):
            v.validate()

    def test_float_var(self):
        """Test FloatVar with min/max bounds and an extra validator."""
        v = FloatVar(
            key="test_float", label="Test", min_val=10.5, max_val=20.5
        )
        v.value = 15.0
        v.validate()

        v.value = 10.4
        with pytest.raises(ValidationError, match="at least 10.5"):
            v.validate()

    def test_slider_float_var(self):
        """Test SliderFloatVar to ensure it behaves like a FloatVar."""
        v = SliderFloatVar(
            key="slider", label="Slider", min_val=0.0, max_val=1.0
        )
        v.value = 0.5
        v.validate()

        v.value = 1.1
        with pytest.raises(ValidationError, match="at most 1.0"):
            v.validate()

    def test_choice_var(self):
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

    @patch("rayforge.core.varset.hostnamevar.is_valid_hostname_or_ip")
    def test_hostname_var(self, mock_is_valid):
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

    def test_serial_port_var(self):
        """Test SerialPortVar validation logic."""
        v = SerialPortVar(key="port", label="Port")
        v.value = "/dev/ttyUSB0"
        v.validate()

        v.value = None
        with pytest.raises(ValidationError, match="cannot be empty"):
            v.validate()

    def test_port_var(self):
        """Test PortVar for network ports, checking its built-in validation."""
        v = PortVar(key="port", label="Network Port")
        v.value = 8080
        v.validate()

        v.value = 0
        with pytest.raises(ValidationError, match="at least 1"):
            v.validate()

    @patch("rayforge.machine.transport.serial.SerialTransport.list_baud_rates")
    def test_baudrate_var(self, mock_list_rates):
        """Test BaudrateVar against a mocked list of standard rates."""
        mock_list_rates.return_value = [9600, 19200, 115200]
        v = BaudrateVar(key="baud", label="Baud Rate")

        v.value = 115200
        v.validate()

        v.value = 9601
        with pytest.raises(ValidationError, match="not a standard baud rate"):
            v.validate()
