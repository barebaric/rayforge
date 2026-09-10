"""Knife step settings pages."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.core.varset import VarSet
from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ..steps.knife_cut_step import KnifeCutStep


class KnifeStepSettingsPage(StepSettingsPage):
    """Base page for knife step settings.

    Shows the step's own knife settings; subclasses override
    ``_add_step_sections``.
    """

    def __init__(self, editor: "DocEditor", step: "KnifeCutStep"):
        super().__init__(editor, step)
        self._add_step_sections()

    def _add_step_sections(self):
        """Add step-specific sections right after the General section."""

    def _knife_group(self) -> VarSet | None:
        """The domain varset group holding the knife settings."""
        groups = self.step.recipe_varset_groups()
        return groups[0][1] if groups else None


class DragKnifePage(KnifeStepSettingsPage):
    """Settings page for the drag knife cut step."""

    def _add_step_sections(self):
        knife_vars = self._knife_group()
        if knife_vars is None:
            return
        self.add_varset_section(
            _("Drag Knife"),
            knife_vars,
            description=_("Cut the workpiece outline with a drag knife."),
        )


class TangentialKnifePage(KnifeStepSettingsPage):
    """Settings page for the tangential knife cut step."""

    def _add_step_sections(self):
        knife_vars = self._knife_group()
        if knife_vars is None:
            return
        self.add_varset_section(
            _("Tangential Knife"),
            knife_vars,
            description=_(
                "Cut the workpiece outline with a rotary knife that "
                "stays tangent to the path."
            ),
        )
