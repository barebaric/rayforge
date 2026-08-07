"""Tests for step-declared recipe keys and recipe varsets.

Verifies that each step's recipe_keys() is composed correctly through
the inheritance hierarchy and that recipe_varset() exposes the keys
needed by the recipe editor.
"""

from cnc_essentials.steps import (
    AdaptiveClearStep,
    FlatSpiralStep,
    HelixPlungeStep,
    ProfileInnerStep,
    ProfileOuterStep,
    RampEntryStep,
    SlotStep,
    ToroidalClearStep,
)
from cnc_essentials.steps.cnc_assembler_step import CncAssemblerStep

from rayforge.core.step import Step


class TestRecipeKeys:
    """recipe_keys() composition through the step hierarchy."""

    def test_base_step_keys(self):
        assert "cut_speed" in Step.recipe_keys()
        assert "travel_speed" in Step.recipe_keys()

    def test_cnc_step_extends_base(self):
        assert set(Step.recipe_keys()).issubset(
            set(CncAssemblerStep.recipe_keys())
        )
        for key in (
            "tool_diameter",
            "spindle_rpm",
            "plunge_speed",
            "target_depth",
            "depth_per_pass",
            "safe_z",
        ):
            assert key in CncAssemblerStep.recipe_keys()

    def test_adaptive_clear_extends_cnc(self):
        assert set(CncAssemblerStep.recipe_keys()).issubset(
            set(AdaptiveClearStep.recipe_keys())
        )
        for key in (
            "step_over",
            "step_length",
            "max_deflection_deg",
            "wall_margin",
            "area_tolerance",
        ):
            assert key in AdaptiveClearStep.recipe_keys()

    def test_profile_inner_extends_cnc(self):
        assert set(CncAssemblerStep.recipe_keys()).issubset(
            set(ProfileInnerStep.recipe_keys())
        )
        for key in ("step_over", "step_length", "wall_margin"):
            assert key in ProfileInnerStep.recipe_keys()

    def test_profile_outer_extends_cnc(self):
        assert set(CncAssemblerStep.recipe_keys()).issubset(
            set(ProfileOuterStep.recipe_keys())
        )
        for key in ("step_over", "step_length", "wall_margin"):
            assert key in ProfileOuterStep.recipe_keys()

    def test_toroidal_clear_extends_cnc(self):
        assert set(CncAssemblerStep.recipe_keys()).issubset(
            set(ToroidalClearStep.recipe_keys())
        )
        assert "step_over" in ToroidalClearStep.recipe_keys()

    def test_simple_steps_inherit_cnc_keys(self):
        """Steps without extra attrs inherit CncAssemblerStep keys."""
        for cls in (
            FlatSpiralStep,
            HelixPlungeStep,
            RampEntryStep,
            SlotStep,
        ):
            assert cls.recipe_keys() == CncAssemblerStep.recipe_keys()


class TestRecipeVarsetKeys:
    """recipe_varset() keys are consistent with recipe_keys().

    The CNC domain varset covers all process keys but not
    ``selected_head_uid`` (same as the base Step pattern).
    """

    def test_cnc_step_varset(self):
        keys = [var.key for var in CncAssemblerStep.recipe_varset()]
        for key in CncAssemblerStep.recipe_keys():
            assert key in keys, f"Missing var for recipe key '{key}'"

    def test_adaptive_clear_varset_covers_keys(self):
        keys = [var.key for var in AdaptiveClearStep.recipe_varset()]
        for key in AdaptiveClearStep.recipe_keys():
            assert key in keys, f"Missing var for recipe key '{key}'"

    def test_profile_inner_varset_covers_keys(self):
        keys = [var.key for var in ProfileInnerStep.recipe_varset()]
        for key in ProfileInnerStep.recipe_keys():
            assert key in keys, f"Missing var for recipe key '{key}'"

    def test_profile_outer_varset_covers_keys(self):
        keys = [var.key for var in ProfileOuterStep.recipe_varset()]
        for key in ProfileOuterStep.recipe_keys():
            assert key in keys, f"Missing var for recipe key '{key}'"

    def test_toroidal_clear_varset_covers_keys(self):
        keys = [var.key for var in ToroidalClearStep.recipe_varset()]
        for key in ToroidalClearStep.recipe_keys():
            assert key in keys, f"Missing var for recipe key '{key}'"


class TestRecipeVarsetGroups:
    """recipe_varset_groups() splits into CNC and Step Settings."""

    def test_cnc_base_single_group(self):
        groups = CncAssemblerStep.recipe_varset_groups()
        assert len(groups) == 1
        assert groups[0][0] == "CNC"

    def test_adaptive_clear_splits(self):
        groups = AdaptiveClearStep.recipe_varset_groups()
        assert len(groups) == 2
        titles = [g[0] for g in groups]
        assert "CNC" in titles
        assert "Step Settings" in titles

    def test_profile_inner_splits(self):
        groups = ProfileInnerStep.recipe_varset_groups()
        assert len(groups) == 2

    def test_profile_outer_splits(self):
        groups = ProfileOuterStep.recipe_varset_groups()
        assert len(groups) == 2

    def test_toroidal_clear_splits(self):
        groups = ToroidalClearStep.recipe_varset_groups()
        assert len(groups) == 2

    def test_simple_steps_single_group(self):
        for cls in (
            FlatSpiralStep,
            HelixPlungeStep,
            RampEntryStep,
            SlotStep,
        ):
            groups = cls.recipe_varset_groups()
            assert len(groups) == 1
