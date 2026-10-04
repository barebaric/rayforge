from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw

from rayforge.ui_gtk.doceditor.post_processor.groups import (
    ExpanderHost,
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.shared.pref_rows import LengthSpinRow

from ..transformers import MeshCorrectionTransformer

if TYPE_CHECKING:
    from rayforge.core.step import Step


class MeshCorrectionSettingsGroup(TransformerSettingsGroup):
    """UI for configuring the MeshCorrectionTransformer.

    The bed height map itself is probed in the machine settings; this
    group only offers the Z offset applied on top of the map. The
    group manages its own availability: when the machine has no mesh,
    it shows a banner pointing at the Bed Mesh page and insensitizes
    its rows.
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

        self._no_mesh_banner = Adw.Banner(
            title=_(
                "No bed mesh has been probed yet. Use the Bed Mesh page "
                "in the machine settings to probe one."
            )
        )
        super().add(self._no_mesh_banner)

        z_offset_row = LengthSpinRow(
            _("Z Offset"),
            _("Added on top of the probed surface height"),
            lower=-100.0,
            upper=100.0,
            value_in_base=transformer.z_offset,
        )
        self.add(z_offset_row)
        self.z_offset_row = z_offset_row
        z_offset_row.value_changed.connect(self._on_z_offset_changed)

        self._update_sensitivity()

    def _has_mesh(self) -> bool:
        from rayforge.context import get_context

        machine = get_context().machine
        return bool(machine and machine.bed_mesh is not None)

    def _update_sensitivity(self) -> None:
        """Gate rows by the switch and the machine's mesh state.

        The group stays visible and interactive (its enable switch can
        still claim a layer entry); only the meaningless controls are
        insensitive while no mesh is probed.
        """
        super()._update_sensitivity()
        enabled = self._is_enabled()
        has_mesh = self._has_mesh()
        self._no_mesh_banner.set_revealed(enabled and not has_mesh)
        self.z_offset_row.set_sensitive(enabled and has_mesh)

    def _on_z_offset_changed(self, row: LengthSpinRow) -> None:
        self.param_changed.send(
            self,
            key="z_offset",
            value=row.get_value_in_base_units(),
            name=_("Change Bed Mesh Z Offset"),
        )
