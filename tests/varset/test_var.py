import unittest
from typing import cast
from unittest.mock import patch
from rayforge.shared.varset.var import Var, ValidationError
from rayforge.shared.varset.intvar import IntVar
from rayforge.shared.varset.floatvar import FloatVar
from rayforge.shared.varset.hostnamevar import HostnameVar
from rayforge.shared.varset.serialportvar import SerialPortVar
from rayforge.shared.varset.portvar import PortVar
from rayforge.shared.varset.baudratevar import BaudrateVar


class TestVar(unittest.TestCase):
    def test_creation_basic(self):
        """Test basic Var creation with a default value."""
        v = Var(key="test", label="Test", var_type=str, default="hello")
        self.assertEqual(v.key, "test")
        self.assertEqual(v.label, "Test")
        self.assertEqual(v.var_type, str)
        self.assertEqual(v.default, "hello")
        self.assertEqual(v.value, "hello")

    def test_creation_with_value(self):
        """Test that an explicit value overrides the default on creation."""
        v = Var(key="test", label="Test", var_type=int, default=10, value=20)
        self.assertEqual(v.value, 20)

    def test_creation_no_default(self):
        """Test Var creation without a default value."""
        v = Var(key="test", label="Test", var_type=int)
        self.assertIsNone(v.default)
        self.assertIsNone(v.value)

    def test_set_value(self):
        """Test setting a value after creation."""
        v = Var(key="test", label="Test", var_type=int, default=10)
        v.value = 50
        self.assertEqual(v.value, 50)
        v.value = 100
        self.assertEqual(v.value, 100)

    def test_type_coercion_numeric(self):
        """Test automatic type coercion for numeric types from strings."""
        # int from string
        v_int = Var(key="test_i", label="Test I", var_type=int)
        v_int.value = cast(int, "123")
        self.assertEqual(v_int.value, 123)
        self.assertIsInstance(v_int.value, int)

        # int from float string
        v_int.value = cast(int, "123.9")
        self.assertEqual(v_int.value, 123)
        self.assertIsInstance(v_int.value, int)

        # float from string
        v_float = Var(key="test_f", label="Test F", var_type=float)
        v_float.value = cast(float, "123.45")
        self.assertEqual(v_float.value, 123.45)
        self.assertIsInstance(v_float.value, float)

    def test_type_coercion_boolean(self):
        """Test automatic type coercion for boolean types."""
        v_bool = Var(key="test_b", label="Test B", var_type=bool)

        # Test true values from string (case-insensitive)
        for s_true in ("true", "1", "on", "yes", "TRUE"):
            v_bool.value = cast(bool, s_true)
            self.assertIs(v_bool.value, True, f"Failed for '{s_true}'")

        # Test false values from string (case-insensitive)
        for s_false in ("false", "0", "off", "no", "FALSE"):
            v_bool.value = cast(bool, s_false)
            self.assertIs(v_bool.value, False, f"Failed for '{s_false}'")

        # Test from int
        v_bool.value = cast(bool, 1)
        self.assertIs(v_bool.value, True)
        v_bool.value = cast(bool, 0)
        self.assertIs(v_bool.value, False)
        v_bool.value = cast(bool, 5)  # any non-zero number is True
        self.assertIs(v_bool.value, True)

    def test_type_mismatch_error(self):
        """Test that a TypeError is raised for invalid coercions."""
        v_int = Var(key="test", label="Test", var_type=int)
        with self.assertRaisesRegex(TypeError, "cannot be coerced"):
            v_int.value = cast(int, "not a number")

        v_bool = Var(key="test_b", label="Test B", var_type=bool)

        with self.assertRaisesRegex(
            TypeError, "cannot be coerced to type bool"
        ):
            v_bool.value = cast(bool, "maybe")

    def test_validation_logic(self):
        """Tests that validation is an explicit step, separate from setting."""

        def range_check(val):
            # A validator might allow None, or check for it.
            if val is None or not (0 <= val <= 100):
                raise ValidationError("Value out of range 0-100")

        v = Var(
            key="test",
            label="Test",
            var_type=int,
            validator=range_check,
            value=50,
        )

        # 1. Set a new, valid value and validate
        v.value = 75
        self.assertEqual(v.value, 75)
        v.validate()  # Should not raise

        # 2. Set an invalid value; the assignment itself should NOT fail
        v.value = 101
        self.assertEqual(
            v.value, 101, "Value should be updated even if invalid"
        )

        # 3. Now, explicit validation should fail
        with self.assertRaisesRegex(
            ValidationError, "Value out of range 0-100"
        ):
            v.validate()

        # 4. The value remains the invalid one after the failed validation
        self.assertEqual(v.value, 101)

    # --- Tests for Var subclasses ---

    def test_int_var(self):
        """Test IntVar with min/max bounds and an extra validator."""
        # Test bounds
        v = IntVar(key="test_int", label="Test", min_val=10, max_val=20)
        v.value = 15
        v.validate()  # Should pass

        v.value = 9
        with self.assertRaisesRegex(ValidationError, "at least 10"):
            v.validate()

        v.value = 21
        with self.assertRaisesRegex(ValidationError, "at most 20"):
            v.validate()

        # Test that None is a valid value if not otherwise constrained
        v.value = None
        v.validate()  # Should pass, as checks are guarded by `v is not None`

        # Test extra validator
        def is_even(v_val):
            if v_val is not None and v_val % 2 != 0:
                raise ValidationError("Must be even")

        v_even = IntVar(
            key="test_even",
            label="Test",
            min_val=0,
            max_val=10,
            validator=is_even,
        )

        v_even.value = 4
        v_even.validate()  # Should pass

        v_even.value = 3
        with self.assertRaisesRegex(ValidationError, "Must be even"):
            v_even.validate()

        # Check that bounds still work with the extra validator
        v_even.value = 12
        with self.assertRaisesRegex(ValidationError, "at most 10"):
            v_even.validate()

    def test_int_var_with_non_nullable_validator(self):
        """Test an IntVar with a validator that rejects None."""

        def not_none(v):
            if v is None:
                raise ValidationError("Value cannot be None")

        v = IntVar(key="test", label="Test", validator=not_none)
        v.value = 10
        v.validate()  # Should pass

        v.value = None
        with self.assertRaisesRegex(ValidationError, "Value cannot be None"):
            v.validate()

    def test_float_var(self):
        """Test FloatVar with min/max bounds and an extra validator."""
        v = FloatVar(
            key="test_float", label="Test", min_val=10.5, max_val=20.5
        )
        v.value = 15.0
        v.validate()  # Should pass

        v.value = 10.4
        with self.assertRaisesRegex(ValidationError, "at least 10.5"):
            v.validate()

        v.value = 20.6
        with self.assertRaisesRegex(ValidationError, "at most 20.5"):
            v.validate()

        # Test that None is a valid value if not otherwise constrained
        v.value = None
        v.validate()  # Should pass, as checks are guarded by `if v is None`

        # Test extra validator
        def is_whole(v_val):
            if v_val is not None and v_val != int(v_val):
                raise ValidationError("Must be a whole number")

        v_whole = FloatVar(
            key="test_whole", label="Test", extra_validator=is_whole
        )
        v_whole.value = 5.0
        v_whole.validate()

        v_whole.value = 5.5
        with self.assertRaisesRegex(ValidationError, "Must be a whole number"):
            v_whole.validate()

    @patch("rayforge.shared.varset.hostnamevar.is_valid_hostname_or_ip")
    def test_hostname_var(self, mock_is_valid):
        """Test HostnameVar validation logic."""
        mock_is_valid.side_effect = lambda h: h in ["valid.com", "1.2.3.4"]
        v = HostnameVar(key="host", label="Host")

        # Test valid hostname
        v.value = "valid.com"
        v.validate()  # Should pass

        # Test valid IP
        v.value = "1.2.3.4"
        v.validate()  # Should pass

        # Test invalid hostname
        v.value = "invalid-hostname"
        with self.assertRaisesRegex(
            ValidationError, "Invalid hostname or IP address format"
        ):
            v.validate()

        # Test empty value (fails validation)
        v.value = ""
        with self.assertRaisesRegex(ValidationError, "cannot be empty"):
            v.validate()

        # Test None value (fails validation)
        v.value = None
        with self.assertRaisesRegex(ValidationError, "cannot be empty"):
            v.validate()

    def test_serial_port_var(self):
        """Test SerialPortVar validation logic."""
        v = SerialPortVar(key="port", label="Port")

        # Test valid value
        v.value = "/dev/ttyUSB0"
        v.validate()  # Should pass

        # Test empty value (fails validation)
        v.value = ""
        with self.assertRaisesRegex(ValidationError, "cannot be empty"):
            v.validate()

        # Test None value (fails validation)
        v.value = None
        with self.assertRaisesRegex(ValidationError, "cannot be empty"):
            v.validate()

    def test_port_var(self):
        """Test PortVar for network ports, checking its built-in validation."""
        v = PortVar(key="port", label="Network Port")

        v.value = 8080
        v.validate()

        v.value = 0
        with self.assertRaisesRegex(ValidationError, "at least 1"):
            v.validate()

        v.value = 65536
        with self.assertRaisesRegex(ValidationError, "at most 65535"):
            v.validate()

        v.value = None
        with self.assertRaisesRegex(ValidationError, "cannot be empty"):
            v.validate()

    @patch("rayforge.machine.transport.serial.SerialTransport.list_baud_rates")
    def test_baudrate_var(self, mock_list_rates):
        """Test BaudrateVar against a mocked list of standard rates."""
        mock_list_rates.return_value = [9600, 19200, 115200]
        v = BaudrateVar(key="baud", label="Baud Rate")

        v.value = 115200
        v.validate()

        v.value = 9601
        with self.assertRaisesRegex(
            ValidationError, "not a standard baud rate"
        ):
            v.validate()

        v.value = None
        with self.assertRaisesRegex(ValidationError, "cannot be empty"):
            v.validate()
