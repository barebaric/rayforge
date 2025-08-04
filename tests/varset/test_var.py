import unittest

from rayforge.varset.var import Var
from rayforge.transport.serial import SerialPort
from rayforge.driver.util import Hostname


class TestVar(unittest.TestCase):
    def test_creation_basic(self):
        v = Var(key="test", label="Test", var_type=str, default="hello")
        self.assertEqual(v.key, "test")
        self.assertEqual(v.label, "Test")
        self.assertEqual(v.var_type, str)
        self.assertEqual(v.default, "hello")
        self.assertEqual(v.value, "hello")

    def test_creation_with_value(self):
        v = Var(key="test", label="Test", var_type=int, default=10, value=20)
        self.assertEqual(v.value, 20)

    def test_creation_no_default(self):
        v = Var(key="test", label="Test", var_type=int)
        self.assertIsNone(v.default)
        self.assertIsNone(v.value)

    def test_set_value(self):
        v = Var(key="test", label="Test", var_type=int, default=10)
        v.value = 50
        self.assertEqual(v.value, 50)
        v.value = 100
        self.assertEqual(v.value, 100)

    def test_type_coercion(self):
        v_int = Var(key="test_i", label="Test I", var_type=int)
        # Test runtime coercion. Tell linter to ignore intentional type
        # mismatch.
        v_int.value = "123"  # type: ignore[assignment]
        self.assertEqual(v_int.value, 123)
        self.assertIsInstance(v_int.value, int)

        v_float = Var(key="test_f", label="Test F", var_type=float)
        v_float.value = "123.45"  # type: ignore[assignment]
        self.assertEqual(v_float.value, 123.45)
        self.assertIsInstance(v_float.value, float)

        v_bool_from_int = Var(key="test_b", label="Test B", var_type=bool)
        v_bool_from_int.value = 1  # type: ignore[assignment]
        self.assertIs(v_bool_from_int.value, True)
        v_bool_from_int.value = 0  # type: ignore[assignment]
        self.assertIs(v_bool_from_int.value, False)

    def test_type_mismatch_error(self):
        v = Var(key="test", label="Test", var_type=int)
        with self.assertRaisesRegex(TypeError, "cannot be coerced"):
            v.value = "not a number"  # type: ignore[assignment]

    def test_custom_types(self):
        v_host = Var(key="host", label="Host", var_type=Hostname)
        v_host.value = "my-device.local"  # type: ignore[assignment]
        # Now that Hostname is a class, we can check for it directly.
        self.assertIsInstance(v_host.value, Hostname)
        self.assertEqual(v_host.value, "my-device.local")

        v_port = Var(key="port", label="Port", var_type=SerialPort)
        v_port.value = "/dev/ttyUSB0"  # type: ignore[assignment]
        # Now that SerialPort is a class, we can check for it directly.
        self.assertIsInstance(v_port.value, SerialPort)
        self.assertEqual(v_port.value, "/dev/ttyUSB0")

    def test_validator_success(self):
        def my_validator(x):
            self.assertIn(x, [10, 20, 30])

        v = Var(key="test", label="Test", var_type=int, validator=my_validator)
        v.value = 10
        v.value = 20
        self.assertEqual(v.value, 20)

    def test_validator_failure(self):
        def range_check(val):
            if not (0 <= val <= 100):
                raise ValueError("Value out of range 0-100")

        v = Var(key="test", label="Test", var_type=int, validator=range_check)
        with self.assertRaisesRegex(ValueError, "Validation failed"):
            v.value = 101
        with self.assertRaisesRegex(ValueError, "Validation failed"):
            v.value = -1

    def test_value_not_updated_on_failure(self):
        def range_check(val):
            if not (0 <= val <= 100):
                raise ValueError("Value out of range 0-100")

        v = Var(key="test", label="Test", var_type=int, validator=range_check)
        v.value = 50
        with self.assertRaises(ValueError):
            v.value = 200
        self.assertEqual(v.value, 50)
