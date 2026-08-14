"""Base for settings groups that manage a transformer."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Protocol

from blinker import Signal
from gi.repository import Adw, GObject, Gtk

from .....pipeline.transformer.base import OpsTransformer
from ....icons import get_icon

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

    Two enable controls are supported:

    * **Step mode** (default): the group builds a plain
      :class:`Gtk.Switch` (exposed as :attr:`enable_switch`) that the
      host page places into the expander row's header; the remaining
      rows are gated by it.
    * **Recipe mode** (``apply_toggle=True``): additionally builds a
      toggle-button prefix. The toggle decides whether the recipe
      applies the transformer at all (``recipe_apply``); the enable
      switch still controls the transformer's own ``enabled``. The
      toggle is exposed as :attr:`apply_toggle` so the host page can
      place it, and new states are announced via the
      :attr:`apply_changed` signal.
    """

    def __init__(
        self,
        title: str,
        transformer: OpsTransformer,
        page: ExpanderHost,
        *,
        step: "Step | None" = None,
        apply_toggle: bool = False,
        initial_apply: bool = False,
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
            apply_toggle: When True, build an apply toggle prefix
                  instead of the enable switch. The switch is still
                  built so the transformer can be enabled/disabled
                  within the recipe.
            initial_apply: The initial state of the apply toggle.
        """
        super().__init__(
            title=title,
            description=transformer.description,
            **kwargs,
        )
        self.param_changed = Signal()
        self.apply_changed = Signal()
        self.transformer = transformer
        self.page = page
        self.step = step
        self._rows: list[Gtk.Widget] = []
        self.enable_switch: Gtk.Switch | None = None
        self.apply_toggle: Gtk.ToggleButton | None = None

        self._add_enable_switch(transformer)
        if apply_toggle:
            self._build_apply_toggle(initial_apply)

    def add(self, child: Gtk.Widget) -> None:
        self._rows.append(child)
        if not self.page.use_expanders:
            super().add(child)
        if self.enable_switch is not None:
            child.set_sensitive(self._is_enabled())

    def _add_enable_switch(self, transformer: OpsTransformer) -> None:
        """Build the enable switch for the group's header.

        The switch is a plain :class:`Gtk.Switch`, not a row: the host
        page places it into the expander row's header (``add_suffix``)
        and reparents the parameter rows itself. The remaining rows are
        gated by this switch.
        """
        switch = Gtk.Switch()
        switch.set_active(transformer.enabled)
        switch.set_valign(Gtk.Align.CENTER)
        switch.set_tooltip_text(_("Enable {}").format(transformer.label))
        switch.connect("notify::active", self._on_enable_toggled)
        self.enable_switch = switch

    def _on_enable_toggled(
        self, switch: Gtk.Switch, _pspec: GObject.ParamSpec
    ) -> None:
        self.param_changed.send(
            self,
            key="enabled",
            value=switch.get_active(),
            name=_("Toggle {}").format(self.transformer.label),
        )
        self._update_sensitivity()

    def _build_apply_toggle(self, initial_apply: bool) -> None:
        """Build the apply toggle prefix for recipe mode."""
        toggle = Gtk.ToggleButton()
        toggle.add_css_class("flat")
        toggle.set_valign(Gtk.Align.CENTER)
        icon = get_icon("check-symbolic")
        icon.set_valign(Gtk.Align.CENTER)
        toggle.set_child(icon)
        toggle.set_active(initial_apply)
        toggle.set_tooltip_text(_("Apply this transformer to the step"))
        toggle.connect("toggled", self._on_apply_toggled)
        self.apply_toggle = toggle
        self._update_apply_visual()

    def _on_apply_toggled(self, toggle: Gtk.ToggleButton) -> None:
        applied = toggle.get_active()
        self.apply_changed.send(self, state=applied)
        self._update_apply_visual()

    def set_apply_state(self, applied: bool) -> None:
        """Set the apply toggle state programmatically (no signal)."""
        if self.apply_toggle is not None:
            self.apply_toggle.set_active(applied)
        self._update_apply_visual()

    def get_apply_state(self) -> bool:
        """Whether the recipe should apply this transformer."""
        return (
            self.apply_toggle.get_active()
            if self.apply_toggle is not None
            else False
        )

    def _update_apply_visual(self) -> None:
        if self.apply_toggle is not None:
            # Dim the group while the toggle is off so users see that
            # the recipe will not apply it. The rows stay interactive,
            # matching the recipe settings rows.
            self.set_opacity(1.0 if self.get_apply_state() else 0.5)

    def _is_enabled(self) -> bool:
        """Whether the rows are currently editable.

        Rows are gated by the enable switch in both modes; the apply
        toggle (recipe mode) only dims the group when off.
        """
        assert self.enable_switch is not None
        return self.enable_switch.get_active()

    def _update_sensitivity(self) -> None:
        """Gate rows by the enable switch, which stays clickable."""
        enabled = self._is_enabled()
        for row in self._rows:
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
