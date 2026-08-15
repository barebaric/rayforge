import pytest

from rayforge import config as config_module
from rayforge.context import get_context
from rayforge.core.recipe import Recipe
from rayforge.core.step import Step
from rayforge.doceditor.step_cmd import StepCmd


@pytest.fixture
def step_cmd(doc_editor):
    """Provides a StepCmd instance."""
    return StepCmd(doc_editor)


def test_set_step_param(step_cmd):
    """Test setting a step parameter."""
    target_dict = {}
    key = "test_key"
    new_value = "test_value"
    name = "Test Command"

    step_cmd.set_step_param(target_dict, key, new_value, name)

    assert target_dict[key] == new_value


def test_set_step_param_no_change(step_cmd):
    """Test that setting the same value does nothing."""
    target_dict = {"test_key": "test_value"}
    key = "test_key"
    new_value = "test_value"
    name = "Test Command"

    step_cmd.set_step_param(target_dict, key, new_value, name)

    assert target_dict[key] == new_value


def test_set_step_param_float_tolerance(step_cmd):
    """Test that setting a float value within tolerance does nothing."""
    target_dict = {"test_key": 1.0}
    key = "test_key"
    new_value = 1.0000001  # Within 1e-6 tolerance
    name = "Test Command"

    step_cmd.set_step_param(target_dict, key, new_value, name)

    assert target_dict[key] == 1.0


class PowerStep(Step):
    """A Step subclass owning one setter-less recipe key."""

    power: float

    @classmethod
    def recipe_keys(cls) -> tuple[str, ...]:
        return ("cut_speed", "power")


@pytest.fixture
def isolated_recipe_mgr(doc_editor, monkeypatch, tmp_path):
    """Point the context's recipe manager at a temp directory."""
    monkeypatch.setattr(
        config_module, "USER_RECIPES_DIR", tmp_path / "recipes"
    )
    get_context()._recipe_mgr = None
    return doc_editor.context.recipe_mgr


def test_apply_best_recipe_to_step_gates_and_uses_setters(
    step_cmd, isolated_recipe_mgr
):
    """Settings are gated through recipe_keys and applied via setters."""
    isolated_recipe_mgr.add_recipe(
        Recipe(
            name="Test Recipe",
            setting_dicts=[
                {"name": "cut_speed", "value": 750.7, "recipe_apply": True},
                {"name": "power", "value": 0.9, "recipe_apply": True},
                {"name": "travel_speed", "value": 1, "recipe_apply": True},
                {"name": "__class__", "value": "X", "recipe_apply": True},
                {"name": "_hidden", "value": 5, "recipe_apply": True},
            ],
        )
    )

    step = PowerStep(typelabel="Test", name="Test")
    step_cmd.apply_best_recipe_to_step(step)

    # cut_speed goes through the base Step setter (int coercion).
    assert step.cut_speed == 750
    # power has no setter, so it is assigned directly.
    assert step.power == 0.9
    # Not owned by PowerStep, dunder and private names: untouched.
    assert step.travel_speed == 5000
    assert step.__class__ is PowerStep
    assert not hasattr(step, "_hidden")
    assert step.applied_recipe_uid is not None


def test_apply_recipe_transformers_skips_undeclared_keys(step_cmd):
    """Only param keys the step's transformer dict declares are written."""
    step = PowerStep(typelabel="Test", name="Test")
    step.per_workpiece_transformers_dicts = [
        {"name": "CropTransformer", "enabled": True, "offset": 0.5}
    ]
    recipe = Recipe(
        transformer_dicts=[
            {
                "name": "CropTransformer",
                "recipe_apply": True,
                "enabled": False,
                "offset": 2.0,
                "foreign_key": 99,
            }
        ]
    )

    step_cmd._apply_recipe_transformers_to_step(step, recipe)

    (d,) = step.per_workpiece_transformers_dicts
    assert d["enabled"] is False
    assert d["offset"] == 2.0
    assert "foreign_key" not in d


def test_set_recipe_value_ignores_unknown_keys():
    """Direct calls with non-recipe keys are no-ops."""
    step = PowerStep(typelabel="Test", name="Test")
    step.set_recipe_value("__class__", "X")
    step.set_recipe_value("_hidden", 5)
    step.set_recipe_value("travel_speed", 1)

    assert step.__class__ is PowerStep
    assert not hasattr(step, "_hidden")
    assert step.travel_speed == 5000


def test_set_recipe_value_prefers_setter():
    """Valid keys apply via the setter when present, else assignment."""
    step = PowerStep(typelabel="Test", name="Test")
    step.set_recipe_value("cut_speed", 750.7)
    assert step.cut_speed == 750  # coerced by set_cut_speed

    step.set_recipe_value("power", 0.9)
    assert step.power == 0.9
