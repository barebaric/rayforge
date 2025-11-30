import pytest
import math
from rayforge.core.sketcher.params import ParameterContext


@pytest.fixture
def params():
    return ParameterContext()


def test_set_get_simple(params):
    params.set("width", 100)
    assert params.get("width") == 100.0


def test_expression_evaluation(params):
    params.set("a", 10)
    params.set("b", 20)
    params.set("c", "a + b")
    assert params.get("c") == 30.0


def test_math_functions(params):
    params.set("x", "sqrt(16)")
    params.set("y", "pi")
    assert params.get("x") == 4.0
    assert params.get("y") == pytest.approx(math.pi)


def test_dependency_resolution_order(params):
    # 'b' depends on 'a', but 'b' is defined first (conceptually)
    # The solver iterates, so order of set() shouldn't strictly matter
    # if evaluate() is called after.
    params.set("b", "a * 2")
    params.set("a", 10)
    assert params.get("b") == 20.0


def test_chained_dependencies(params):
    params.set("val1", 10)
    params.set("val2", "val1 + 5")  # 15
    params.set("val3", "val2 * 2")  # 30
    assert params.get("val3") == 30.0


def test_dirty_flag_logic(params):
    params.set("x", 10)
    assert params.get("x") == 10.0
    # Modifying a dependency should mark dirty and re-eval
    params.set("x", 20)
    params.set("y", "x + 5")
    assert params.get("y") == 25.0


def test_missing_dependency_safe_fail(params):
    # Should not crash, returns 0.0 or stays unresolved
    params.set("z", "non_existent + 5")
    assert params.get("z") == 0.0


def test_evaluate_arbitrary_string(params):
    params.set("w", 50)
    result = params.evaluate("w / 2")
    assert result == 25.0


def test_circular_dependency_protection(params):
    """
    Test that circular dependencies don't cause infinite recursion/hanging.
    """
    params.set("a", "b")
    params.set("b", "a")

    # This should evaluate without crashing (likely returning 0.0 or failing
    # resolution). The current implementation limits passes to len(exprs),
    # so it is safe.
    assert params.get("a") == 0.0
    assert params.get("b") == 0.0


def test_parameter_syntax_error(params):
    """Test graceful handling of bad math strings."""
    params.set("bad", "sqrt(")  # Incomplete syntax
    assert params.get("bad") == 0.0


def test_parameter_overwrite(params):
    """Test overwriting a parameter updates dependents."""
    params.set("base", 10)
    params.set("res", "base * 2")
    assert params.get("res") == 20.0

    params.set("base", 5)
    assert params.get("res") == 10.0


def test_parameter_context_serialization_round_trip(params):
    """Tests to_dict and from_dict for ParameterContext."""
    params.set("width", 100)
    params.set("height", "width / 2")
    params.set("depth", "sqrt(width)")

    data = params.to_dict()
    assert data == {
        "expressions": {
            "width": "100",
            "height": "width / 2",
            "depth": "sqrt(width)",
        }
    }

    new_params = ParameterContext.from_dict(data)

    # Check that expressions were loaded and evaluate correctly
    assert new_params.get("width") == 100.0
    assert new_params.get("height") == 50.0
    assert new_params.get("depth") == 10.0
