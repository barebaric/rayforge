from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw, GObject

from rayforge.ui_gtk.doceditor.post_processor.groups import (
    ExpanderHost,
    TransformerSettingsGroup,
)

from ..transformers import Optimize

if TYPE_CHECKING:
    from rayforge.core.step import Step


class OptimizeSettingsGroup(TransformerSettingsGroup):
    """UI for configuring the Optimize transformer."""

    def __init__(
        self,
        title: str,
        transformer: Optimize,
        page: ExpanderHost,
        *,
        step: "Step | None" = None,
        **kwargs,
    ):
        super().__init__(title, transformer, page, step=step, **kwargs)

        self.flip_row = Adw.SwitchRow(
            title=_("Allow Flipping"),
            subtitle=_("Allow reversing path direction for shorter travel"),
        )
        self.flip_row.set_active(transformer.allow_flip)
        self.add(self.flip_row)
        self.flip_row.connect("notify::active", self._on_flip_toggled)

        self.preserve_row = Adw.SwitchRow(
            title=_("Preserve First Workpiece"),
            subtitle=_("Keep the first workpiece at its original position"),
        )
        self.preserve_row.set_active(transformer.preserve_first)
        self.add(self.preserve_row)
        self.preserve_row.connect(
            "notify::active", self._on_preserve_first_toggled
        )

    def _on_flip_toggled(
        self, row: Adw.SwitchRow, _pspec: GObject.ParamSpec
    ) -> None:
        self.param_changed.send(
            self,
            key="allow_flip",
            value=row.get_active(),
            name=_("Toggle Flipping"),
        )

    def _on_preserve_first_toggled(
        self, row: Adw.SwitchRow, _pspec: GObject.ParamSpec
    ) -> None:
        self.param_changed.send(
            self,
            key="preserve_first",
            value=row.get_active(),
            name=_("Toggle Preserve First Workpiece"),
        )
