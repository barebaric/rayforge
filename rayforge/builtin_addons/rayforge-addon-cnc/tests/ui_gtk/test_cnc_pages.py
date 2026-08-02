# flake8: noqa: E402
"""UI tests for the CNC step settings pages."""

from typing import Any

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from cnc_essentials.widgets.pages import ProfileOuterPage

from rayforge.core.step_registry import step_registry
from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage
from rayforge.ui_gtk.doceditor.step_settings.rows import (
    CutSpeedRow,
    TravelSpeedRow,
)


@pytest.mark.ui
def test_profile_outer_page_composes_common_sections(
    editor, cnc_machine, ui_context
):
    step_cls = step_registry.get("ProfileOuterStep")
    assert step_cls is not None
    step: Any = step_cls.create(ui_context)

    page = ProfileOuterPage(editor, step)
    assert isinstance(page, StepSettingsPage)

    rows = list(page._rows)
    assert any(isinstance(row, CutSpeedRow) for row in rows)
    assert any(isinstance(row, TravelSpeedRow) for row in rows)
