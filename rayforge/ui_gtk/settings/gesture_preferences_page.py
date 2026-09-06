"""Settings page for configuring mouse gestures."""

import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...context import get_context
from ..gestures import GestureSlot, gesture_registry
from ..shared.preferences_page import TrackedPreferencesPage
from .gesture_capture import GestureBindingPopover

logger = logging.getLogger(__name__)


class GesturePreferencesPage(TrackedPreferencesPage):
    """
    Preferences page showing the mouse gesture bindings of all
    registered gesture contexts.

    The page is rebuilt whenever the gesture registry changes, so
    addon-contributed contexts appear and disappear live.
    """

    key = "gestures"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_title(_("Mouse Gestures"))
        self.set_icon_name("input-mouse-symbolic")
        self._rows: dict[tuple[str, str], Adw.ActionRow] = {}
        self._buttons: dict[tuple[str, str], Gtk.Button] = {}
        self._groups: list[Adw.PreferencesGroup] = []
        self._config = get_context().config
        self._config.changed.connect(self._on_config_changed)
        gesture_registry.changed.connect(self._on_registry_changed)
        self.connect("destroy", self._on_destroyed)
        self._rebuild()

    def _on_destroyed(self, widget) -> None:
        self._config.changed.disconnect(self._on_config_changed)
        gesture_registry.changed.disconnect(self._on_registry_changed)

    def _rebuild(self) -> None:
        for group in self._groups:
            self.remove(group)
        self._groups.clear()
        self._rows.clear()
        self._buttons.clear()
        for context in gesture_registry.get_contexts():
            slots = gesture_registry.get_slots(context.id)
            if not slots:
                continue
            group = Adw.PreferencesGroup(
                title=context.label, description=context.description
            )
            for slot in slots:
                group.add(self._build_slot_row(context.id, slot))
            self.add(group)
            self._groups.append(group)

    def _build_slot_row(
        self, context_id: str, slot: GestureSlot
    ) -> Adw.ActionRow:
        row = Adw.ActionRow(title=slot.label)
        subtitle_parts = []
        if slot.description:
            subtitle_parts.append(slot.description)
        if slot.default_binding is not None:
            subtitle_parts.append(
                _("Default: %s") % slot.default_binding.display_label()
            )
        if subtitle_parts:
            row.set_subtitle("\n".join(subtitle_parts))

        button = Gtk.Button(valign=Gtk.Align.CENTER)
        button.connect("clicked", self._on_edit_binding, context_id, slot)
        row.add_suffix(button)

        self._rows[(context_id, slot.id)] = row
        self._buttons[(context_id, slot.id)] = button
        self._update_row(context_id, slot)
        return row

    def _update_row(self, context_id: str, slot: GestureSlot) -> None:
        binding = gesture_registry.resolve_binding(
            self._config, context_id, slot.id
        )
        if binding is not None:
            label = binding.display_label()
        else:
            label = _("Unassigned")
        button = self._buttons.get((context_id, slot.id))
        if button is not None:
            button.set_label(label)

    def _on_edit_binding(
        self, button: Gtk.Button, context_id: str, slot: GestureSlot
    ) -> None:
        popover = GestureBindingPopover(
            self._config, context_id, slot, on_applied=self._refresh_rows
        )
        popover.set_parent(button)
        popover.popdown()
        popover.present()
        popover.connect("closed", lambda p: p.unparent())

    def _refresh_rows(self) -> None:
        for context_id, slot_id in self._rows:
            slot = gesture_registry.get_slot(context_id, slot_id)
            if slot is not None:
                self._update_row(context_id, slot)

    def _on_config_changed(self, sender, **kwargs) -> None:
        self._refresh_rows()

    def _on_registry_changed(self, sender, **kwargs) -> None:
        self._rebuild()
