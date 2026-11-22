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

    def _step_settings_group(self) -> VarSet | None:
        """The varset group holding the remaining step settings."""
        groups = self.step.recipe_varset_groups()
        return groups[1][1] if len(groups) > 1 else None

    def _add_common_sections(self, title: str, description: str) -> None:
        """Add the knife section and the remaining step settings."""
        knife_vars = self._knife_group()
        if knife_vars is None:
            return
        self.add_varset_section(title, knife_vars, description=description)
        step_vars = self._step_settings_group()
        if step_vars is not None:
            self.add_varset_section(_("Step Settings"), step_vars)


class DragKnifePage(KnifeStepSettingsPage):
    """Settings page for the drag knife cut step."""

    def _add_step_sections(self):
        self._add_common_sections(
            _("Drag Knife"),
            _("Cut the workpiece outline with a drag knife."),
        )


class TangentialKnifePage(KnifeStepSettingsPage):
    """Settings page for the tangential knife cut step."""

    def _add_step_sections(self):
        self._add_common_sections(
            _("Tangential Knife"),
            _(
                "Cut the workpiece outline with a rotary knife that "
                "stays tangent to the path."
            ),
        )
