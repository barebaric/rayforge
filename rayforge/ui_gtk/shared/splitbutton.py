import logging
from typing import Sequence, Tuple
from gi.repository import Gtk
from ..icons import get_icon

logger = logging.getLogger(__name__)


class SplitMenuButton(Gtk.Box):
    """
    A composite widget that mimics a split button, integrated with Gio.Action.

    It has a main action button that shows and triggers the last-used action,
    and a separate dropdown button to reveal all other actions in a popover.
    """

    def __init__(
        self,
        actions: Sequence[Tuple[str, str, str]],
        default_index: int = 0,
        **kwargs,
    ):
        """
        Initializes the SplitMenuButton.

        Args:
            actions: A sequence of tuples, where each tuple contains
                     (name, icon_name, action_name) for an action.
            default_index: The index of the action to show by default.
        """
        super().__init__(**kwargs)
        self.set_orientation(Gtk.Orientation.HORIZONTAL)
        self.set_spacing(0)
        self.add_css_class("linked")

        if not actions:
            raise ValueError("SplitMenuButton requires at least one action.")

        self.actions = actions
        self._last_action_index = default_index

        # 1. The main action button
        self.main_button = Gtk.Button()
        # The action will be set dynamically by _set_active_action
        self.append(self.main_button)

        # 2. The dropdown button for the menu
        popover = self._build_popover()
        self.menu_button = Gtk.MenuButton(
            child=get_icon("pan-down-symbolic"),
            popover=popover,
            tooltip_text=_("Show all options"),
        )
        self.append(self.menu_button)

        # Set the initial state of the main button
        self._set_active_action(self._last_action_index)

    def set_sensitive(self, sensitive: bool):
        """Sets the sensitivity of the entire composite button."""
        # The sensitivity of the buttons is now controlled by their associated
        # Gio.Actions. This method can still be used for a top-level override.
        super().set_sensitive(sensitive)

    def _build_popover(self) -> Gtk.Popover:
        """Creates the popover menu with buttons for all actions."""
        popover = Gtk.Popover()
        list_box = Gtk.ListBox()
        list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        list_box.add_css_class("popover-list")
        popover.set_child(list_box)

        for i, (name, icon_name, action_name) in enumerate(self.actions):
            row = Gtk.ListBoxRow()
            button = Gtk.Button()
            button.set_has_frame(False)
            content = Gtk.Box(spacing=6)
            content.append(get_icon(icon_name))
            content.append(Gtk.Label(label=name))
            button.set_child(content)

            # Set the action for the menu item button
            button.set_action_name(action_name)

            # Also connect to 'clicked' to update the main button's appearance
            button.connect(
                "clicked", lambda _, idx=i: self._on_menu_item_clicked(idx)
            )
            row.set_child(button)
            list_box.append(row)

        return popover

    def _on_menu_item_clicked(self, index: int):
        """
        Called when a user clicks an item in the popover menu.
        This updates the main button to reflect the choice.
        """
        popover = self.menu_button.get_popover()
        if popover:
            popover.popdown()
        self._set_active_action(index)

    def _set_active_action(self, index: int):
        """
        Updates the main button to show the new active action's icon
        and connects it to the correct Gio.Action.

        Args:
            index: The index of the action to set as active.
        """
        self._last_action_index = index
        name, icon_name, action_name = self.actions[index]

        # Update the main button's appearance and its active action
        self.main_button.set_child(get_icon(icon_name))
        self.main_button.set_tooltip_text(name)
        self.main_button.set_action_name(action_name)
