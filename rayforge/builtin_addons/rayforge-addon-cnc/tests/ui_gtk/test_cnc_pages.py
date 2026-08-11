# flake8: noqa: E402
"""UI tests for the CNC step settings pages."""

from typing import cast

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from cnc_essentials.steps.cnc_assembler_step import CncAssemblerStep
from cnc_essentials.widgets.pages import AdaptiveClearPage, ProfileOuterPage
from cnc_essentials.widgets.rows import (
    MaxDeflectionRow,
    PlungeSpeedRow,
    ToolDiameterRow,
)

from rayforge.core.step_registry import step_registry
from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage
from rayforge.ui_gtk.doceditor.step_settings.rows import (
    CutSpeedRow,
    TravelSpeedRow,
)
from rayforge.ui_gtk.shared.pref_rows import (
    AngleSpinRow,
    LengthSpinRow,
    SpeedSpinRow,
)


def _find(widget, cls):
    for row in widget._rows:
        if isinstance(row, cls):
            return row
    raise AssertionError(f"row {cls.__name__} not found in page")


@pytest.mark.ui
def test_profile_outer_page_composes_common_sections(
    editor, cnc_machine, ui_context
):
    step_cls = step_registry.get("ProfileOuterStep")
    assert step_cls is not None
    step = cast(CncAssemblerStep, step_cls.create(ui_context))

    page = ProfileOuterPage(editor, step)
    assert isinstance(page, StepSettingsPage)

    rows = list(page._rows)
    assert any(isinstance(row, CutSpeedRow) for row in rows)
    assert any(isinstance(row, TravelSpeedRow) for row in rows)


@pytest.mark.ui
def test_length_rows_use_user_units(editor, cnc_machine, ui_context):
    ui_context.config.unit_preferences["length"] = "in"
    step_cls = step_registry.get("ProfileOuterStep")
    assert step_cls is not None
    step = cast(CncAssemblerStep, step_cls.create(ui_context))

    page = ProfileOuterPage(editor, step)
    tool = _find(page, ToolDiameterRow)
    assert isinstance(tool.widget, LengthSpinRow)

    step.tool_diameter = 25.4
    step.updated.send(step)

    assert tool.widget.get_value_in_base_units() == pytest.approx(25.4)
    assert tool.widget.get_value() == pytest.approx(1.0, abs=1e-2)


@pytest.mark.ui
def test_plunge_speed_row_uses_speed_units(editor, cnc_machine, ui_context):
    step_cls = step_registry.get("ProfileOuterStep")
    assert step_cls is not None
    step = cast(CncAssemblerStep, step_cls.create(ui_context))

    page = ProfileOuterPage(editor, step)
    plunge = _find(page, PlungeSpeedRow)
    assert isinstance(plunge.widget, SpeedSpinRow)


@pytest.mark.ui
def test_deflection_row_uses_angle_spin_row(editor, cnc_machine, ui_context):
    step_cls = step_registry.get("AdaptiveClearStep")
    assert step_cls is not None
    step = cast(CncAssemblerStep, step_cls.create(ui_context))

    page = AdaptiveClearPage(editor, step)
    deflection = _find(page, MaxDeflectionRow)
    assert isinstance(deflection.widget, AngleSpinRow)
