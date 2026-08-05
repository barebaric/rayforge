"""Wavefront step settings widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow

from .rows import LaserStepSettingsPage

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class StepOverRow(SpinRow):
    """Spin row bound to the ``step_over_mm`` attribute."""

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(
            editor,
            step,
            "step_over_mm",
            _("Step Over"),
            _("Lateral step-over between wavefront passes"),
            0.05,
            50.0,
            0.1,
            2,
            quantity="length",
        )


class OffsetRow(SpinRow):
    """Spin row bound to the ``offset_mm`` attribute."""

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(
            editor,
            step,
            "offset_mm",
            _("Offset"),
            _("Extra offset from walls"),
            0.0,
            20.0,
            0.1,
            2,
            quantity="length",
        )


class WavefrontStepSettingsPage(LaserStepSettingsPage):
    """Settings page for the WavefrontStep."""

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(editor, step)
        self.add_section(
            _("Wavefront"),
            StepOverRow,
            OffsetRow,
            description=_("Clear pockets with a wavefront toolpath."),
        )
