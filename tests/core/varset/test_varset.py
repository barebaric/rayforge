import pytest
from unittest.mock import Mock
from rayforge.core.varset.var import Var, ValidationError
from rayforge.core.varset.intvar import IntVar
from rayforge.core.varset.floatvar import FloatVar
from rayforge.core.varset.choicevar import ChoiceVar
from rayforge.core.varset.varset import VarSet


class TestVarSet:
    def test_creation(self):
        """Test basic creation of an empty VarSet."""
        vs = VarSet(title="My Settings", description="Some settings.")
        assert vs.title == "My Settings"
        assert vs.description == "Some settings."
        assert len(vs) == 0

    def test_creation_with_vars(self):
        """Test creating a VarSet pre-populated with Var objects."""
        vars_list = [
            Var(key="a", label="A", var_type=str),
            Var(key="b", label="B", var_type=int),
        ]
        vs = VarSet(vars=vars_list)
        assert len(vs) == 2
        assert "a" in vs.keys()
        assert "b" in vs.keys()

    def test_add_var(self):
        """Test adding a Var to the set."""
        vs = VarSet()
        v = Var(key="test1", label="Test 1", var_type=str, default="abc")
        vs.add(v)
        assert len(vs) == 1
        assert "test1" in vs.keys()
        assert vs["test1"] is v

    def test_add_duplicate_key(self):
        """Test that adding a Var with a duplicate key raises a KeyError."""
        vs = VarSet()
        v1 = Var(key="test1", label="Test 1", var_type=str)
        v2 = Var(key="test1", label="Test 2", var_type=int)
        vs.add(v1)
        with pytest.raises(KeyError):
            vs.add(v2)

    def test_get_and_keys(self):
        """Test getting Vars by key and retrieving all keys."""
        vs = VarSet()
        v1 = Var(key="a", label="A", var_type=str)
        vs.add(v1)
        assert vs.get("a") is v1
        assert vs.get("nonexistent") is None
        assert list(vs.keys()) == ["a"]

    def test_set_value_by_key(self):
        """Test setting a Var's value using dictionary-style access."""
        vs = VarSet()
        vs.add(Var(key="timeout", label="Timeout", var_type=int, default=10))
        vs["timeout"] = 30
        assert vs["timeout"].value == 30

    def test_set_value_nonexistent_key(self):
        """
        Test that setting a value for a nonexistent key raises a KeyError.
        """
        vs = VarSet()
        with pytest.raises(KeyError):
            vs["nonexistent"] = 100

    def test_iteration_and_vars_property(self):
        """Test iteration, insertion order, and the .vars property."""
        vs = VarSet()
        v1 = Var(key="b_var", label="B", var_type=str)
        v2 = Var(key="a_var", label="A", var_type=str)
        v3 = Var(key="c_var", label="C", var_type=str)
        vs.add(v1)
        vs.add(v2)
        vs.add(v3)

        iterated_vars = list(vs)
        assert len(iterated_vars) == 3
        assert iterated_vars[0] is v1
        assert iterated_vars[1] is v2
        assert iterated_vars[2] is v3

        property_vars = vs.vars
        assert isinstance(property_vars, list)
        assert property_vars == iterated_vars

    def test_len(self):
        """Test the __len__ method."""
        vs = VarSet()
        assert len(vs) == 0
        vs.add(Var(key="a", label="A", var_type=str))
        assert len(vs) == 1

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
        assert values == expected

    def test_set_values(self):
        """
        Test the set_values method for updating multiple Vars from a dict.
        """
        vs = VarSet()
        vs.add(Var(key="name", label="Name", var_type=str, default="ray"))
        vs.add(Var(key="speed", label="Speed", var_type=int, value=1000))

        new_values = {"name": "forge", "speed": 2000, "extra": "ignore"}
        vs.set_values(new_values)

        assert vs["name"].value == "forge"
        assert vs["speed"].value == 2000

    def test_clear(self):
        """Test the clear method to remove all Vars."""
        vs = VarSet()
        vs.add(Var(key="name", label="Name", var_type=str, default="ray"))
        assert len(vs) == 1
        vs.clear()
        assert len(vs) == 0

    def test_validate(self):
        """Test the master validate method on the VarSet."""
        vs = VarSet()
        v_ok = Var(key="ok", label="OK", var_type=str, value="good")
        v_bad = IntVar(key="bad", label="Bad", min_val=10, max_val=20, value=5)
        vs.add(v_ok)
        vs.add(v_bad)

        with pytest.raises(ValidationError, match="at least 10"):
            vs.validate()

        vs["bad"] = 15
        vs.validate()  # Should not raise

    def test_repr(self):
        """Test the __repr__ method."""
        vs = VarSet(title="My Settings")
        vs.add(Var(key="a", label="A", var_type=str))
        vs.add(Var(key="b", label="B", var_type=int))
        representation = repr(vs)
        assert "title='My Settings'" in representation
        assert "count=2" in representation

    def test_serialization_round_trip(self):
        """
        Test a full serialization/deserialization cycle of a complex VarSet.
        """
        original_vs = VarSet(
            title="Complex Settings", description="A mix of var types."
        )
        original_vs.add(
            IntVar(key="count", label="Count", default=5, min_val=0)
        )
        original_vs.add(
            FloatVar(key="factor", label="Factor", default=1.2, max_val=2.0)
        )
        original_vs.add(
            ChoiceVar(
                key="mode",
                label="Mode",
                choices=["A", "B"],
                default="A",
            )
        )

        serialized_data = original_vs.to_dict()
        rehydrated_vs = VarSet.from_dict(serialized_data)

        assert rehydrated_vs.title == original_vs.title
        assert rehydrated_vs.description == original_vs.description
        assert len(rehydrated_vs) == len(original_vs)

        for key in original_vs.keys():
            assert rehydrated_vs[key].to_dict() == original_vs[key].to_dict()

    # --- Observability Signal Tests ---

    def test_var_added_signal(self):
        """Test that the var_added signal is emitted correctly."""
        vs = VarSet()
        listener = Mock()
        vs.var_added.connect(listener)

        v1 = Var(key="a", label="A", var_type=str)
        vs.add(v1)

        listener.assert_called_once_with(vs, var=v1)

    def test_var_removed_signal(self):
        """Test that the var_removed signal is emitted correctly."""
        vs = VarSet()
        v1 = Var(key="a", label="A", var_type=str)
        vs.add(v1)

        listener = Mock()
        vs.var_removed.connect(listener)

        vs.remove("a")
        listener.assert_called_once_with(vs, var=v1)

        # Removing a non-existent key should not trigger the signal
        listener.reset_mock()
        vs.remove("non-existent")
        listener.assert_not_called()

    def test_cleared_signal(self):
        """Test that the cleared signal is emitted correctly."""
        vs = VarSet()
        vs.add(Var(key="a", label="A", var_type=str))
        listener = Mock()
        vs.cleared.connect(listener)

        vs.clear()
        listener.assert_called_once_with(vs)

    def test_var_value_changed_bubble_up_signal(self):
        """
        Test that value changes in child Vars bubble up through the VarSet.
        """
        vs = VarSet()
        v1 = IntVar(key="a", label="A", value=10)
        vs.add(v1)

        listener = Mock()
        vs.var_value_changed.connect(listener)

        # Change value via dictionary access on the VarSet
        vs["a"] = 20
        listener.assert_called_once_with(
            vs, var=v1, new_value=20, old_value=10
        )

        # Change value directly on the Var object
        listener.reset_mock()
        v1.value = 30
        listener.assert_called_once_with(
            vs, var=v1, new_value=30, old_value=20
        )

    def test_disconnect_on_remove_and_clear(self):
        """Test that signal listeners are disconnected on remove and clear."""
        vs = VarSet()
        v1 = IntVar(key="a", label="A", value=10)
        vs.add(v1)

        listener = Mock()
        vs.var_value_changed.connect(listener)

        # 1. Test remove()
        vs.remove("a")
        v1.value = 50  # Change value of the now-removed var
        listener.assert_not_called()  # The VarSet should not be listening

        # 2. Test clear()
        v2 = IntVar(key="b", label="B", value=100)
        vs.add(v2)
        vs.clear()
        v2.value = 200  # Change value of the now-cleared var
        listener.assert_not_called()
