import logging
from typing import List, Optional
from gi.repository import Gtk  # type: ignore
from .. import HistoryManager
from ..models.command import Command


logger = logging.getLogger(__name__)

# The maximum number of history items to display in the dropdown.
HISTORY_DISPLAY_LIMIT = 15


class _HistoryButton(Gtk.Box):
    """
    A composite widget that combines a main action button with a dropdown
    button for history. This is the base class for Undo/Redo buttons.
    The main button is controlled by a Gio.Action.
    """

    def __init__(self, icon_name: str, tooltip: str, **kwargs):
        super().__init__(**kwargs)
        self.set_orientation(Gtk.Orientation.HORIZONTAL)
        self.set_spacing(0)
        # The "linked" style class makes the two buttons appear joined
        # together.
        self.get_style_context().add_class("linked")

        self.manager: Optional[HistoryManager] = None

        # 1. The main action button
        self.main_button = Gtk.Button(icon_name=icon_name)
        self.main_button.set_tooltip_text(tooltip)
        self.append(self.main_button)

        # 2. The dropdown button for history
        self.menu_button = Gtk.MenuButton(icon_name="pan-down-symbolic")
        self.menu_button.set_tooltip_text(_("Show History"))
        popover = Gtk.Popover()
        self.menu_button.set_popover(popover)
        self.append(self.menu_button)

    def set_action_name(self, action_name: str):
        """Sets the Gio.Action for the main button."""
        self.main_button.set_action_name(action_name)

    def set_history_manager(self, manager: HistoryManager):
        """Connects the button to a HistoryManager instance."""
        if self.manager:
            self.manager.changed.disconnect(self._on_history_changed)

        self.manager = manager
        self.manager.changed.connect(self._on_history_changed)
        self._on_history_changed(self.manager)

    def _on_history_changed(self, sender: HistoryManager, **kwargs):
        """Updates the button's state and menu when the history changes."""
        # The main_button's sensitivity is now controlled by its Gio.Action.
        # We only need to control the dropdown arrow's sensitivity here.
        can_act = self._can_act()
        self.menu_button.set_sensitive(can_act)

        popover = self.menu_button.get_popover()

        # Get the full stack and then limit it for display purposes.
        full_stack = self._get_stack()
        display_stack = full_stack[:HISTORY_DISPLAY_LIMIT]

        if display_stack:
            # Rebuild the listbox only if needed
            if not popover.get_child() or not isinstance(
                popover.get_child(), Gtk.ListBox
            ):
                list_box = Gtk.ListBox()
                list_box.set_selection_mode(Gtk.SelectionMode.NONE)
                list_box.get_style_context().add_class("popover-list")
                popover.set_child(list_box)

            list_box = popover.get_child()
            # A simple implementation is to clear and refill.
            # For high-frequency updates, one might optimize this.
            child = list_box.get_first_child()
            while child:
                list_box.remove(child)
                child = list_box.get_first_child()

            for command in display_stack:
                row = Gtk.ListBoxRow()
                button = Gtk.Button(label=command.name or _("Unnamed Action"))
                button.set_has_frame(False)
                button.connect("clicked", self._on_menu_item_clicked, command)
                row.set_child(button)
                list_box.append(row)
        else:
            if popover.get_child():
                popover.set_child(None)

    def _on_menu_item_clicked(self, _, command: Command):
        """Performs undo/redo up to a specific command from the popover."""
        self.menu_button.get_popover().popdown()
        if not self.manager:
            return
        self._act_to(command)

    def _get_stack(self) -> List[Command]:
        """Subclasses must implement this to return the correct stack."""
        raise NotImplementedError

    def _can_act(self) -> bool:
        """
        Subclasses must implement this to check if an action can be performed.
        """
        raise NotImplementedError

    def _act_to(self, command: Command):
        """Subclasses must implement this for the history dropdown action."""
        raise NotImplementedError


class UndoButton(_HistoryButton):
    """A Gtk.Box composite widget for undoing actions."""

    def __init__(self, **kwargs):
        super().__init__(
            icon_name="edit-undo-symbolic",
            tooltip=_("Undo the last action"),
            **kwargs,
        )

    def _get_stack(self) -> List[Command]:
        if not self.manager:
            return []
        # Newest action should appear first in the dropdown.
        return list(reversed(self.manager.undo_stack))

    def _can_act(self) -> bool:
        return self.manager is not None and self.manager.can_undo()

    def _act_to(self, command: Command):
        if not self.manager:
            return
        self.manager.undo_to(command)


class RedoButton(_HistoryButton):
    """A Gtk.Box composite widget for redoing actions."""

    def __init__(self, **kwargs):
        super().__init__(
            icon_name="edit-redo-symbolic",
            tooltip=_("Redo the last action"),
            **kwargs,
        )

    def _get_stack(self) -> List[Command]:
        if not self.manager:
            return []
        # Newest action to be redone should appear first.
        return list(reversed(self.manager.redo_stack))

    def _can_act(self) -> bool:
        return self.manager is not None and self.manager.can_redo()

    def _act_to(self, command: Command):
        if not self.manager:
            return
        self.manager.redo_to(command)
