"""Base for settings groups that manage a transformer."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Protocol

from blinker import Signal
from gi.repository import Adw, GObject, Gtk

from .....pipeline.transformer.base import OpsTransformer

if TYPE_CHECKING:
    from .....core.step import Step


class ExpanderHost(Protocol):
    """A page that wraps transformer groups in expander rows.

    The group defers to the host: when ``use_expanders`` is set, the
    host extracts the group's rows from ``_rows`` and reparents them
    itself, so the group must not add them to its own hierarchy.
    """

    use_expanders: bool = True


class TransformerSettingsGroup(Adw.PreferencesGroup):
    """
    Base class for settings groups managing a post-processing
    transformer.

    The group is a pure UI widget: it renders the transformer's
    parameters from the :class:`OpsTransformer` instance it is given
    and announces user changes via the :attr:`param_changed` signal.
    It never writes to an editor, history manager, or backing dict —
    the host page decides how to persist the announced changes.

    An enable/disable switch is added automatically and the remaining
    rows are gated by it.
    """

    def __init__(
        self,
        title: str,
        transformer: OpsTransformer,
        page: ExpanderHost,
        *,
        step: "Step | None" = None,
        **kwargs,
    ):
        """
        Args:
            title: The title for the preferences group.
            transformer: The OpsTransformer instance this group configures.
            page: The host page. When it uses expanders, rows are only
                  tracked in :attr:`_rows` for the host to reparent.
            step: Optional Step object used as read-only context (e.g.
                  for auto-distance calculation). ``None`` in recipe
                  mode.
        """
        super().__init__(
            title=title,
            description=transformer.description,
            **kwargs,
        )
        self.param_changed = Signal()
        self.transformer = transformer
        self.page = page
        self.step = step
        self._rows: list[Gtk.Widget] = []
        self.enable_switch: Adw.SwitchRow | None = None

        self._add_enable_switch(transformer)

    def add(self, child: Gtk.Widget) -> None:
        self._rows.append(child)
        if not self.page.use_expanders:
            super().add(child)
        if self.enable_switch is not None and child is not self.enable_switch:
            child.set_sensitive(self.enable_switch.get_active())

    def _add_enable_switch(self, transformer: OpsTransformer) -> None:
        switch_row = Adw.SwitchRow(
            title=_("Enable {}").format(transformer.label),
        )
        switch_row.set_active(transformer.enabled)
        self.add(switch_row)
        self.enable_switch = switch_row
        switch_row.connect("notify::active", self._on_enable_toggled)

    def _on_enable_toggled(
        self, row: Adw.SwitchRow, _pspec: GObject.ParamSpec
    ) -> None:
        self.param_changed.send(
            self,
            key="enabled",
            value=row.get_active(),
            name=_("Toggle {}").format(self.transformer.label),
        )
        self._update_sensitivity()

    def _update_sensitivity(self) -> None:
        assert self.enable_switch is not None
        enabled = self.enable_switch.get_active()
        for row in self._rows[1:]:
            row.set_sensitive(enabled)

    def is_unsupported(self) -> bool:
        """
        Whether this transformer is enabled but cannot take effect on
        the active machine (e.g. the driver handles the feature
        itself).

        Subclasses override this to flag expander-level warnings. Returns
        False by default.
        """
        return False
