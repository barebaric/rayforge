from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.ui_gtk.doceditor.post_processor.groups import (
    ExpanderHost,
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.shared.pref_rows import LengthSpinRow

from ..transformers import MeshCorrectionTransformer

if TYPE_CHECKING:
    from rayforge.core.step import Step


class MeshCorrectionSettingsGroup(TransformerSettingsGroup):
    """UI for the MeshCorrectionTransformer.

    The bed height map itself is probed in the machine settings; this
    group offers the focus offset applied on top of the probed surface.
    Its group description reports availability — when the machine has
    no probed mesh it says so instead of the transformer's description.
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
        self.z_offset_row = LengthSpinRow(
            _("Focus Offset"),
            _("Distance above the probed surface for the focus point"),
            lower=-100.0,
            upper=100.0,
            value_in_base=transformer.z_offset,
        )
        self.add(self.z_offset_row)
        self.z_offset_row.value_changed.connect(self._on_z_offset_changed)
        self._update_description()
        self._update_sensitivity()

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

    def _update_sensitivity(self) -> None:
        super()._update_sensitivity()
        if hasattr(self, "z_offset_row"):
            self.z_offset_row.set_sensitive(
                self._is_enabled() and self._has_mesh()
            )

    def _on_z_offset_changed(self, row: LengthSpinRow) -> None:
        self.param_changed.send(
            self,
            key="z_offset",
            value=row.get_value_in_base_units(),
            name=_("Change Bed Mesh Focus Offset"),
        )
