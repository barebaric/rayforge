from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw, GObject

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

        self.subdivide_row = Adw.SwitchRow(
            title=_("Subdivide at Grid Lines"),
            subtitle=_(
                "Split straight cuts at the mesh grid lines so Z follows "
                "the bed between probe points. This increases the number "
                "of commands, most noticeably for raster engraving."
            ),
        )
        self.subdivide_row.set_active(transformer.subdivide)
        self.add(self.subdivide_row)
        self.subdivide_row.connect(
            "notify::active", self._on_subdivide_toggled
        )

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
            sensitive = self._is_enabled() and self._has_mesh()
            self.z_offset_row.set_sensitive(sensitive)
            self.subdivide_row.set_sensitive(sensitive)

    def _on_z_offset_changed(self, row: LengthSpinRow) -> None:
        self.param_changed.send(
            self,
            key="z_offset",
            value=row.get_value_in_base_units(),
            name=_("Change Bed Mesh Focus Offset"),
        )

    def _on_subdivide_toggled(
        self, row: Adw.SwitchRow, _pspec: GObject.ParamSpec
    ) -> None:
        self.param_changed.send(
            self,
            key="subdivide",
            value=row.get_active(),
            name=_("Toggle Mesh Subdivision"),
        )
