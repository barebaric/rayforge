"""Tests for step-declared recipe keys and recipe varsets.

Verifies that each step's RECIPE_KEYS is composed correctly through the
inheritance hierarchy and that recipe_varset() exposes the keys needed
by the recipe editor.
"""

from laser_essentials.steps import (
    ContourStep,
    EngraveStep,
    FrameStep,
    ShrinkWrapStep,
)
from laser_essentials.steps.laser_step import LaserStep

from rayforge.core.step import Step


class TestRecipeKeys:
    """RECIPE_KEYS composition through the step hierarchy."""

    def test_base_step_keys(self):
        assert "selected_head_uid" in Step.RECIPE_KEYS
        assert "cut_speed" in Step.RECIPE_KEYS
        assert "travel_speed" in Step.RECIPE_KEYS

    def test_laser_step_extends_base(self):
        assert set(Step.RECIPE_KEYS).issubset(set(LaserStep.RECIPE_KEYS))
        for key in ("power", "air_assist", "kerf_mm", "tab_power"):
            assert key in LaserStep.RECIPE_KEYS

    def test_contour_step_extends_laser(self):
        assert set(LaserStep.RECIPE_KEYS).issubset(
            set(ContourStep.RECIPE_KEYS)
        )
        for key in (
            "cut_side",
            "cut_order",
            "remove_inner_paths",
            "path_offset_mm",
            "overcut",
        ):
            assert key in ContourStep.RECIPE_KEYS

    def test_engrave_step_extends_laser(self):
        assert set(LaserStep.RECIPE_KEYS).issubset(
            set(EngraveStep.RECIPE_KEYS)
        )
        for key in (
            "scan_angle",
            "depth_mode",
            "invert",
            "min_power_level",
            "max_power_level",
        ):
            assert key in EngraveStep.RECIPE_KEYS

    def test_frame_step_extends_laser(self):
        assert set(LaserStep.RECIPE_KEYS).issubset(set(FrameStep.RECIPE_KEYS))
        for key in ("cut_side", "path_offset_mm"):
            assert key in FrameStep.RECIPE_KEYS

    def test_shrinkwrap_step_extends_laser(self):
        assert set(LaserStep.RECIPE_KEYS).issubset(
            set(ShrinkWrapStep.RECIPE_KEYS)
        )
        for key in ("cut_side", "path_offset_mm", "gravity"):
            assert key in ShrinkWrapStep.RECIPE_KEYS


class TestRecipeVarsetKeys:
    """recipe_varset() keys are consistent with RECIPE_KEYS.

    The base Step varset is domain-neutral and does not render the
    head picker, so it only covers the motion keys. Laser steps add the
    laser-domain head var, so their varset covers the full RECIPE_KEYS.
    """

    def test_base_step_varset(self):
        keys = [var.key for var in Step.recipe_varset()]
        assert "cut_speed" in keys
        assert "travel_speed" in keys

    def test_laser_step_varset_includes_head(self):
        keys = [var.key for var in LaserStep.recipe_varset()]
        assert "selected_head_uid" in keys
        for key in ("power", "air_assist", "kerf_mm", "tab_power"):
            assert key in keys

    def test_contour_step_varset_covers_step_keys(self):
        keys = [var.key for var in ContourStep.recipe_varset()]
        for key in ContourStep.RECIPE_KEYS:
            assert key in keys, f"Missing var for recipe key '{key}'"

    def test_engrave_step_varset_covers_step_keys(self):
        keys = [var.key for var in EngraveStep.recipe_varset()]
        for key in EngraveStep.RECIPE_KEYS:
            assert key in keys, f"Missing var for recipe key '{key}'"

    def test_frame_step_varset_covers_step_keys(self):
        keys = [var.key for var in FrameStep.recipe_varset()]
        for key in FrameStep.RECIPE_KEYS:
            assert key in keys, f"Missing var for recipe key '{key}'"

    def test_shrinkwrap_step_varset_covers_step_keys(self):
        keys = [var.key for var in ShrinkWrapStep.recipe_varset()]
        for key in ShrinkWrapStep.RECIPE_KEYS:
            assert key in keys, f"Missing var for recipe key '{key}'"
