"""Settings page for configuring mouse gestures."""

import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...context import get_context
from ..gestures import GestureKind, GestureSlot, gesture_registry
from ..gestures.model import GestureSpec, get_candidate_specs
from ..shared.preferences_page import TrackedPreferencesPage

logger = logging.getLogger(__name__)


class GesturePreferencesPage(TrackedPreferencesPage):
    """
    Preferences page showing the mouse gesture bindings of all
    registered gesture contexts as dropdown rows.

    The page is rebuilt whenever the gesture registry or the bindings
    change, so addon-contributed contexts appear and disappear live and
    gestures already used by another slot of the same context are not
    offered again.
    """

    key = "gestures"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_title(_("Mouse Gestures"))
        self.set_icon_name("input-mouse-symbolic")
        self._rows: dict[tuple[str, str], Adw.ComboRow] = {}
        self._options: dict[tuple[str, str], list[GestureSpec | None]] = {}
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
        self._options.clear()
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
    ) -> Adw.ComboRow:
        row = Adw.ComboRow(title=slot.label)
        subtitle_parts = []
        if slot.description:
            subtitle_parts.append(slot.description)
        if slot.default_binding is not None:
            subtitle_parts.append(
                _("Default: %s") % slot.default_binding.display_label()
            )
        if subtitle_parts:
            row.set_subtitle("\n".join(subtitle_parts))

        options = self._build_options(context_id, slot)
        row.set_model(
            Gtk.StringList.new(
                [self._option_label(option) for option in options]
            )
        )
        row.set_selected(self._selected_index(context_id, slot, options))
        row.connect(
            "notify::selected", self._on_binding_selected, context_id, slot
        )

        self._rows[(context_id, slot.id)] = row
        self._options[(context_id, slot.id)] = options
        return row

    def _build_options(
        self, context_id: str, slot: GestureSlot
    ) -> list[GestureSpec | None]:
        """
        Returns the gesture options the user can pick for a slot: every
        physical gesture of the slot's kind that is not already used by
        another slot of the same context, plus the slot's current
        binding and, if allowed, the "unassigned" choice.
        """
        taken = set()
        for other in gesture_registry.get_slots(context_id):
            if other.id == slot.id:
                continue
            taken.add(
                gesture_registry.resolve_binding(
                    self._config, context_id, other.id
                )
            )
        default = slot.default_binding
        kind = default.kind if default is not None else GestureKind.DRAG
        options: list[GestureSpec | None] = [
            spec
            for spec in get_candidate_specs(kind, buttons=slot.buttons)
            if spec not in taken
        ]
        binding = gesture_registry.resolve_binding(
            self._config, context_id, slot.id
        )
        if binding is not None and binding not in options:
            options.insert(0, binding)
        if slot.allow_unassign:
            options.append(None)
        return options

    @staticmethod
    def _option_label(option: GestureSpec | None) -> str:
        if option is None:
            return _("Unassigned")
        return option.display_label()

    def _selected_index(
        self,
        context_id: str,
        slot: GestureSlot,
        options: list[GestureSpec | None],
    ) -> int:
        binding = gesture_registry.resolve_binding(
            self._config, context_id, slot.id
        )
        try:
            return options.index(binding)
        except ValueError:
            return 0

    def _on_binding_selected(
        self,
        row: Adw.ComboRow,
        _pspec,
        context_id: str,
        slot: GestureSlot,
    ) -> None:
        options = self._options.get((context_id, slot.id))
        if not options:
            return
        index = row.get_selected()
        if index < 0 or index >= len(options):
            return
        option = options[index]
        default = slot.default_binding
        if option is None:
            self._config.set_gesture_binding(context_id, slot.id, None)
        elif default is not None and option == default:
            self._config.reset_gesture_binding(context_id, slot.id)
        else:
            self._config.set_gesture_binding(
                context_id, slot.id, option.to_config_string()
            )
        value = option.to_config_string() if option is not None else "None"
        logger.debug(f"Gesture '{context_id}/{slot.id}' set to {value}")

    def _on_config_changed(self, sender, **kwargs) -> None:
        self._rebuild()

    def _on_registry_changed(self, sender, **kwargs) -> None:
        self._rebuild()
