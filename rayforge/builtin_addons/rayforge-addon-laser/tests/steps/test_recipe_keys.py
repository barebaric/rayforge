"""Tests for step-declared recipe keys and recipe varsets.

Verifies that each step's recipe_keys() is composed correctly through
the inheritance hierarchy and that recipe_varset() exposes the keys
needed by the recipe editor.
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
    """recipe_keys() composition through the step hierarchy."""

    def test_base_step_keys(self):
        assert "cut_speed" in Step.recipe_keys()
        assert "travel_speed" in Step.recipe_keys()

    def test_laser_step_extends_base(self):
        assert set(Step.recipe_keys()).issubset(set(LaserStep.recipe_keys()))
        for key in ("power", "air_assist", "tab_power"):
            assert key in LaserStep.recipe_keys()

    def test_contour_step_extends_laser(self):
        assert set(LaserStep.recipe_keys()).issubset(
            set(ContourStep.recipe_keys())
        )
        for key in (
            "cut_side",
            "cut_order",
            "remove_inner_paths",
            "offset_mm",
            "overcut",
        ):
            assert key in ContourStep.recipe_keys()

    def test_engrave_step_extends_laser(self):
        assert set(LaserStep.recipe_keys()).issubset(
            set(EngraveStep.recipe_keys())
        )
        for key in (
            "scan_angle",
            "depth_mode",
            "invert",
            "min_power_level",
            "max_power_level",
        ):
            assert key in EngraveStep.recipe_keys()

    def test_frame_step_extends_laser(self):
        assert set(LaserStep.recipe_keys()).issubset(
            set(FrameStep.recipe_keys())
        )
        for key in ("cut_side", "offset_mm"):
            assert key in FrameStep.recipe_keys()

    def test_shrinkwrap_step_extends_laser(self):
        assert set(LaserStep.recipe_keys()).issubset(
            set(ShrinkWrapStep.recipe_keys())
        )
        for key in ("cut_side", "offset_mm", "gravity"):
            assert key in ShrinkWrapStep.recipe_keys()


class TestRecipeVarsetKeys:
    """recipe_varset() keys are consistent with recipe_keys().

    The base Step varset is domain-neutral and does not render the
    head picker, so it only covers the motion keys. Laser steps add the
    laser-domain head var, so their varset covers the full recipe keys.
    """

    def test_base_step_varset(self):
        keys = [var.key for var in Step.recipe_varset()]
        assert "cut_speed" in keys
        assert "travel_speed" in keys

    def test_laser_step_varset_includes_head(self):
        keys = [var.key for var in LaserStep.recipe_varset()]
        assert "selected_head_uid" in keys
        for key in ("power", "air_assist", "tab_power"):
            assert key in keys

    def test_contour_step_varset_covers_step_keys(self):
        keys = [var.key for var in ContourStep.recipe_varset()]
        for key in ContourStep.recipe_keys():
            assert key in keys, f"Missing var for recipe key '{key}'"

    def test_engrave_step_varset_covers_step_keys(self):
        keys = [var.key for var in EngraveStep.recipe_varset()]
        for key in EngraveStep.recipe_keys():
            assert key in keys, f"Missing var for recipe key '{key}'"

    def test_frame_step_varset_covers_step_keys(self):
        keys = [var.key for var in FrameStep.recipe_varset()]
        for key in FrameStep.recipe_keys():
            assert key in keys, f"Missing var for recipe key '{key}'"

    def test_shrinkwrap_step_varset_covers_step_keys(self):
        keys = [var.key for var in ShrinkWrapStep.recipe_varset()]
        for key in ShrinkWrapStep.recipe_keys():
            assert key in keys, f"Missing var for recipe key '{key}'"


class TestRecipeVarsetBehavior:
    """Varset rows mirror the step settings dialog behavior."""

    def test_contour_step_rows_match_dialog_order(self):
        """The contour dialog adds CutSideRow, OffsetRow, CutOrderRow,
        RemoveInnerPathsRow, OvercutRow, RescanContentRow,
        ThresholdRow (shown only when rescanning)."""
        keys = [var.key for var in ContourStep.recipe_varset()]
        expected = [
            "cut_side",
            "offset_mm",
            "cut_order",
            "remove_inner_paths",
            "overcut",
            "override_threshold",
            "threshold",
        ]
        assert keys[-len(expected) :] == expected

    def test_cut_side_has_no_none_option(self):
        """The dialog's cut-side combo has no "None" entry, so the var
        must not offer one either."""
        from rayforge.core.varset import LabeledChoiceVar

        var = next(
            v for v in ContourStep.recipe_varset() if v.key == "cut_side"
        )
        assert isinstance(var, LabeledChoiceVar)
        assert var.allow_none is False

    def test_offset_sensitive_when_not_centerline(self):
        """The offset row is insensitive while cut_side is CENTERLINE."""
        var = next(
            v for v in ContourStep.recipe_varset() if v.key == "offset_mm"
        )
        assert var.sensitive_when is not None
        assert var.sensitive_when({"cut_side": "CENTERLINE"}) is False
        assert var.sensitive_when({"cut_side": "OUTSIDE"}) is True

    def test_laser_step_rows_match_dialog_order(self):
        """The laser section order is head, power, speeds, air assist,
        tab power, frequency, pulse width."""
        keys = [var.key for var in LaserStep.recipe_varset()]
        expected = [
            "selected_head_uid",
            "power",
            "cut_speed",
            "travel_speed",
            "air_assist",
            "tab_power",
            "frequency",
            "pulse_width",
        ]
        assert keys == expected
