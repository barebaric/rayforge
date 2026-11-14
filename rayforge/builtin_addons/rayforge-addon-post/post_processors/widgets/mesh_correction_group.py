from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.ui_gtk.doceditor.post_processor.groups import (
    ExpanderHost,
    TransformerSettingsGroup,
)

from ..transformers import MeshCorrectionTransformer

if TYPE_CHECKING:
    from rayforge.core.step import Step


class MeshCorrectionSettingsGroup(TransformerSettingsGroup):
    """UI for the MeshCorrectionTransformer.

    The transformer has no parameters: it applies exactly what the
    probed bed mesh defines. The group's only state is the enable
    switch, whose group description reports availability — when the
    machine has no probed mesh it says so instead of the
    transformer's description.
    """

    def __init__(
        self,
        title: str,
        transformer: MeshCorrectionTransformer,
        page: ExpanderHost,
        *,
        step: "Step | None" = None,
        **kwargs,
    ):
        super().__init__(title, transformer, page, step=step, **kwargs)
        self._update_description()

    def _has_mesh(self) -> bool:
        from rayforge.context import get_context

        machine = get_context().machine
        return bool(machine and machine.bed_mesh is not None)

    def _update_description(self) -> None:
        if self._has_mesh():
            self.set_description(self.transformer.description)
        else:
            self.set_description(
                _(
                    "No bed mesh has been probed yet. Probe one on the "
                    "Bed Mesh page in the machine settings."
                )
            )
