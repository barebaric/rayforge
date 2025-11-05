import unittest
from rayforge.shared.varset.var import Var, ValidationError
from rayforge.shared.varset.intvar import IntVar
from rayforge.shared.varset.varset import VarSet


class TestVarSet(unittest.TestCase):
    def test_creation(self):
        """Test basic creation of an empty VarSet."""
        vs = VarSet(title="My Settings", description="Some settings.")
        self.assertEqual(vs.title, "My Settings")
        self.assertEqual(vs.description, "Some settings.")
        self.assertEqual(len(vs), 0)

    def test_creation_with_vars(self):
        """Test creating a VarSet pre-populated with Var objects."""
        vars_list = [
            Var(key="a", label="A", var_type=str),
            Var(key="b", label="B", var_type=int),
        ]
        vs = VarSet(vars=vars_list)
        self.assertEqual(len(vs), 2)
        self.assertIn("a", vs.keys())
        self.assertIn("b", vs.keys())

    def test_add_var(self):
        """Test adding a Var to the set."""
        vs = VarSet()
        v = Var(key="test1", label="Test 1", var_type=str, default="abc")
        vs.add(v)
        self.assertEqual(len(vs), 1)
        self.assertIn("test1", vs.keys())
        self.assertIs(vs["test1"], v)

    def test_add_duplicate_key(self):
        """Test that adding a Var with a duplicate key raises a KeyError."""
        vs = VarSet()
        v1 = Var(key="test1", label="Test 1", var_type=str)
        v2 = Var(key="test1", label="Test 2", var_type=int)
        vs.add(v1)
        with self.assertRaises(KeyError):
            vs.add(v2)

    def test_get_and_keys(self):
        """Test getting Vars by key and retrieving all keys."""
        vs = VarSet()
        v1 = Var(key="a", label="A", var_type=str)
        vs.add(v1)
        self.assertIs(vs.get("a"), v1)
        self.assertIsNone(vs.get("nonexistent"))
        self.assertEqual(list(vs.keys()), ["a"])

    def test_set_value_by_key(self):
        """Test setting a Var's value using dictionary-style access."""
        vs = VarSet()
        vs.add(Var(key="timeout", label="Timeout", var_type=int, default=10))
        vs["timeout"] = 30
        self.assertEqual(vs["timeout"].value, 30)

    def test_set_value_nonexistent_key(self):
        """
        Test that setting a value for a nonexistent key raises a KeyError.
        """
        vs = VarSet()
        with self.assertRaises(KeyError):
            vs["nonexistent"] = 100

    def test_iteration(self):
        """Test that iterating over a VarSet yields Vars in insertion order."""
        vs = VarSet()
        v1 = Var(key="b_var", label="B", var_type=str)
        v2 = Var(key="a_var", label="A", var_type=str)
        v3 = Var(key="c_var", label="C", var_type=str)
        vs.add(v1)
        vs.add(v2)
        vs.add(v3)

        iterated_vars = list(vs)
        self.assertEqual(len(iterated_vars), 3)
        self.assertIs(iterated_vars[0], v1)
        self.assertIs(iterated_vars[1], v2)
        self.assertIs(iterated_vars[2], v3)

    def test_len(self):
        """Test the __len__ method."""
        vs = VarSet()
        self.assertEqual(len(vs), 0)
        vs.add(Var(key="a", label="A", var_type=str))
        self.assertEqual(len(vs), 1)

    def test_get_values(self):
        """
        Test the get_values method for returning a dict of current values.
        """
        vs = VarSet()
        vs.add(Var(key="name", label="Name", var_type=str, default="ray"))
        vs.add(Var(key="speed", label="Speed", var_type=int, value=1000))
        vs.add(
            Var(key="enabled", label="Enabled", var_type=bool, default=True)
        )
        vs.add(Var(key="empty", label="Empty", var_type=str))

        values = vs.get_values()
        expected = {
            "name": "ray",
            "speed": 1000,
            "enabled": True,
            "empty": None,
        }
        self.assertDictEqual(values, expected)

    def test_set_values(self):
        """
        Test the set_values method for updating multiple Vars from a dict.
        """
        vs = VarSet()
        vs.add(Var(key="name", label="Name", var_type=str, default="ray"))
        vs.add(Var(key="speed", label="Speed", var_type=int, value=1000))
        vs.add(
            Var(key="enabled", label="Enabled", var_type=bool, default=True)
        )

        new_values = {
            "name": "forge",
            "speed": 2000,
            "enabled": False,
            "extra_key": "ignore me",
        }
        vs.set_values(new_values)

        self.assertEqual(vs["name"].value, "forge")
        self.assertEqual(vs["speed"].value, 2000)
        self.assertEqual(vs["enabled"].value, False)

    def test_clear(self):
        """Test the clear method to remove all Vars."""
        vs = VarSet()
        vs.add(Var(key="name", label="Name", var_type=str, default="ray"))
        self.assertEqual(len(vs), 1)
        vs.clear()
        self.assertEqual(len(vs), 0)

    def test_validate(self):
        """Test the master validate method on the VarSet."""
        vs = VarSet()
        v_ok = Var(key="ok", label="OK", var_type=str, value="good")
        v_bad = IntVar(key="bad", label="Bad", min_val=10, max_val=20, value=5)
        vs.add(v_ok)
        vs.add(v_bad)

        # Validation should fail because v_bad's value is out of bounds
        with self.assertRaisesRegex(ValidationError, "at least 10"):
            vs.validate()

        # After fixing the bad value, validation should pass
        vs["bad"] = 15
        try:
            vs.validate()
        except ValidationError:
            self.fail("VarSet.validate() raised ValidationError unexpectedly.")
