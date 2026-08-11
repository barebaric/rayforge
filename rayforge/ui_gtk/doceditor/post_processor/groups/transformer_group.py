"""Base for settings groups that manage a transformer."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Protocol

from blinker import Signal
from gi.repository import Adw, GObject, Gtk

from .....pipeline.transformer.base import OpsTransformer
from ....icons import get_icon
from ....shared.gtk import apply_css

if TYPE_CHECKING:
    from .....core.step import Step

# Make the tri-state menu button look like a plain label + arrow (like
# an Adw.ComboRow) instead of a clickable button: no background, border,
# or hover/focus highlight. The theme styles the internal ``button``
# child of the menubutton node, so the rule must target it.
_APPLY_MENU_CSS = """
.recipe-apply-menu button,
.recipe-apply-menu button:hover,
.recipe-apply-menu button:active,
.recipe-apply-menu button:checked,
.recipe-apply-menu button:focus,
.recipe-apply-menu button:focus-visible,
.recipe-apply-menu button:focus-within {
    background-color: transparent;
    border: none;
    box-shadow: none;
}
"""


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

    * **Step mode** (default): an enable/disable switch is added as the
      first row and the remaining rows are gated by it.
    * **Tri-state mode** (``tri_state=True``): a menu button with three
      options (unchanged / enabled / disabled) replaces the switch. The
      button is exposed as :attr:`tri_state_button` so the host page can
      place it as an expander suffix. The new state is announced via the
      :attr:`tri_state_changed` signal; the page decides what the states
      mean for its backing store.
    """

    #: Tri-state states.
    STATE_UNCHANGED = 0
    STATE_ENABLED = 1
    STATE_DISABLED = 2

    def __init__(
        self,
        title: str,
        transformer: OpsTransformer,
        page: ExpanderHost,
        *,
        step: "Step | None" = None,
        tri_state: bool = False,
        initial_state: int | None = None,
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
            tri_state: When True, build a tri-state apply button instead
                  of an enable switch.
            initial_state: The initial tri-state (one of the
                  :attr:`STATE_*` constants). Defaults to enabled/
                  disabled based on ``transformer.enabled``.
        """
        super().__init__(
            title=title,
            description=transformer.description,
            **kwargs,
        )
        self.param_changed = Signal()
        self.tri_state_changed = Signal()
        self.transformer = transformer
        self.page = page
        self.step = step
        self._rows: list[Gtk.Widget] = []
        self.enable_switch: Adw.SwitchRow | None = None
        self.tri_state_button: Gtk.MenuButton | None = None
        self._tri_state_label: Gtk.Label | None = None
        self._tri_state = self.STATE_UNCHANGED

        if tri_state:
            if initial_state is None:
                initial_state = (
                    self.STATE_ENABLED
                    if transformer.enabled
                    else self.STATE_DISABLED
                )
            self._add_tri_state(transformer, initial_state)
        else:
            self._add_enable_switch(transformer)

    def add(self, child: Gtk.Widget) -> None:
        self._rows.append(child)
        if not self.page.use_expanders:
            super().add(child)
        control = self.tri_state_button or self.enable_switch
        if control is not None and child is not control:
            child.set_sensitive(self._is_enabled())

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

    def _add_tri_state(
        self, transformer: OpsTransformer, initial_state: int
    ) -> None:
        """Build the tri-state apply control."""
        self._tri_state = initial_state

        labels = self._tri_state_labels()
        label_widget = Gtk.Label(label=labels[initial_state])
        self._tri_state_label = label_widget
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        button_box.append(label_widget)
        button_box.append(get_icon("pan-down-symbolic"))

        menu_button = Gtk.MenuButton()
        apply_css(_APPLY_MENU_CSS)
        menu_button.add_css_class("flat")
        menu_button.add_css_class("recipe-apply-menu")
        menu_button.set_child(button_box)

        popover = Gtk.Popover()
        list_box = Gtk.ListBox()
        list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        list_box.add_css_class("popover-list")
        popover.set_child(list_box)
        menu_button.set_popover(popover)

        for state, label in enumerate(labels):
            row = Gtk.ListBoxRow()
            row_button = Gtk.Button(label=label)
            row_button.set_has_frame(False)
            row_button.set_hexpand(True)
            row_button.connect(
                "clicked",
                lambda _b, s=state: (
                    self._on_tri_state_selected(s),
                    popover.popdown(),
                ),
            )
            row.set_child(row_button)
            list_box.append(row)

        self.tri_state_button = menu_button

    @staticmethod
    def _tri_state_labels() -> tuple[str, str, str]:
        return (_("Leave Unchanged"), _("Enabled"), _("Disabled"))

    def _on_tri_state_selected(self, state: int) -> None:
        """Apply a tri-state selection and announce the change."""
        self._tri_state = state
        if self._tri_state_label is not None:
            self._tri_state_label.set_label(self._tri_state_labels()[state])
        self.tri_state_changed.send(self, state=state)
        self._update_sensitivity()

    def get_tri_state(self) -> int:
        """The current tri-state (one of the :attr:`STATE_*` constants)."""
        return self._tri_state

    def _is_enabled(self) -> bool:
        """Whether the enable control currently enables the transformer.

        In tri-state mode only the ``STATE_ENABLED`` state counts as
        enabled; the other states gate the rows off.
        """
        if self.tri_state_button is not None:
            return self._tri_state == self.STATE_ENABLED
        assert self.enable_switch is not None
        return self.enable_switch.get_active()

    def _update_sensitivity(self) -> None:
        enabled = self._is_enabled()
        for row in self._rows:
            if row is not self.enable_switch:
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
