import unittest
from rayforge.varset.var import Var
from rayforge.varset.varset import VarSet


class TestVarSet(unittest.TestCase):
    def test_creation(self):
        vs = VarSet(title="My Settings", description="Some settings.")
        self.assertEqual(vs.title, "My Settings")
        self.assertEqual(vs.description, "Some settings.")
        self.assertEqual(len(vs), 0)

    def test_creation_with_vars(self):
        vars_list = [
            Var(key="a", label="A", var_type=str),
            Var(key="b", label="B", var_type=int),
        ]
        vs = VarSet(vars=vars_list)
        self.assertEqual(len(vs), 2)
        self.assertIn("a", vs._vars)
        self.assertIn("b", vs._vars)

    def test_add_var(self):
        vs = VarSet()
        v = Var(key="test1", label="Test 1", var_type=str, default="abc")
        vs.add(v)
        self.assertEqual(len(vs), 1)
        self.assertIn("test1", vs._vars)
        self.assertIs(vs.get("test1"), v)
        self.assertIs(vs["test1"], v)

    def test_add_duplicate_key(self):
        vs = VarSet()
        v1 = Var(key="test1", label="Test 1", var_type=str)
        v2 = Var(key="test1", label="Test 2", var_type=int)
        vs.add(v1)
        with self.assertRaises(KeyError):
            vs.add(v2)

    def test_set_value_by_key(self):
        vs = VarSet()
        vs.add(Var(key="timeout", label="Timeout", var_type=int, default=10))
        vs["timeout"] = 30
        self.assertEqual(vs["timeout"].value, 30)

    def test_set_value_nonexistent_key(self):
        vs = VarSet()
        with self.assertRaises(KeyError):
            vs["nonexistent"] = 100

    def test_iteration(self):
        vs = VarSet()
        v1 = Var(key="b_var", label="B", var_type=str)
        v2 = Var(key="a_var", label="A", var_type=str)
        v3 = Var(key="c_var", label="C", var_type=str)
        vs.add(v1)
        vs.add(v2)
        vs.add(v3)

        iterated_vars = list(vs)
        self.assertEqual(len(iterated_vars), 3)

        # Check if they are in insertion order
        self.assertIs(iterated_vars[0], v1)  # b_var
        self.assertIs(iterated_vars[1], v2)  # a_var
        self.assertIs(iterated_vars[2], v3)  # c_var

    def test_get_values(self):
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
        vs = VarSet()
        vs.add(Var(key="name", label="Name", var_type=str, default="ray"))
        self.assertEqual(len(vs), 1)
        vs.clear()
        self.assertEqual(len(vs), 0)
