"""Contour step settings widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from rayforge.core.cut_side import CutOrder
from rayforge.ui_gtk.doceditor.step_settings.rows import (
    ComboRow,
    SliderRow,
    SpinRow,
    SwitchRow,
)

from .rows import CutSideRow, LaserStepSettingsPage, OffsetRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class CutOrderRow(ComboRow):
    """Combo row bound to the ``cut_order`` attribute."""

    def __init__(self, editor: "DocEditor", step: Any):
        choices = [(co.label(), co.name) for co in CutOrder]
        super().__init__(
            editor,
            step,
            "cut_order",
            _("Cut Order"),
            choices,
            _("Processing order for nested paths"),
        )


class RemoveInnerPathsRow(SwitchRow):
    """Switch row bound to the ``remove_inner_paths`` attribute."""

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(
            editor,
            step,
            "remove_inner_paths",
            _("Remove Inner Paths"),
            _("If enabled, only trace the outer outline of shapes"),
        )


class OvercutRow(SpinRow):
    """Spin row bound to the ``overcut`` attribute."""

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(
            editor,
            step,
            "overcut",
            _("Overcut"),
            _(
                "Extend closed contours past their start point "
                "so the cut overlaps itself"
            ),
            0.0,
            100.0,
            0.1,
            2,
        )


class RescanContentRow(SwitchRow):
    """Switch row bound to the ``override_threshold`` attribute."""

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(
            editor,
            step,
            "override_threshold",
            _("Rescan Content"),
            _("Ignore source geometry and re-trace within the workpiece"),
        )


class ThresholdRow(SliderRow):
    """Slider row bound to the ``threshold`` attribute.

    Only visible while rescanning content is enabled.
    """

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(
            editor,
            step,
            "threshold",
            _("Tracing Threshold"),
            _("Brightness level (0.0-1.0) to define edges"),
            0.0,
            1.0,
            0.01,
            2,
        )

    def _sync_dependencies(self):
        self.set_visible(self.step.override_threshold)


class ContourStepSettingsPage(LaserStepSettingsPage):
    """Settings page for the ContourStep."""

    include_tab_power = True

    def _add_step_sections(self):
        self.add_section(
            _("Contour Settings"),
            CutSideRow,
            OffsetRow,
            CutOrderRow,
            RemoveInnerPathsRow,
            OvercutRow,
            ThresholdRow,
            RescanContentRow,
            description=_("Trace the outline of the selected shapes."),
        )
