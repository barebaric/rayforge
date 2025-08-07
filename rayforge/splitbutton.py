import logging
from typing import List, Tuple
from gi.repository import Gtk  # type: ignore
from blinker import Signal
from .icons import get_icon


logger = logging.getLogger(__name__)


class SplitMenuButton(Gtk.Box):
    """
    A composite widget that mimics a split button.

    It has a main action button that shows the last-used action and a
    separate dropdown button to reveal all other actions in a popover.
    """

    def __init__(
        self,
        actions: List[Tuple[str, str, Signal]],
        default_index: int = 0,
        **kwargs,
    ):
        """
        Initializes the _SplitMenuButton.

        Args:
            actions: A list of tuples, where each tuple contains
                     (name, icon_name, signal) for an action.
            default_index: The index of the action to show by default.
        """
        super().__init__(**kwargs)
        self.set_orientation(Gtk.Orientation.HORIZONTAL)
        self.set_spacing(0)
        self.get_style_context().add_class("linked")

        self.actions = actions
        self._last_action_index = default_index

        # 1. The main action button
        self.main_button = Gtk.Button()
        self.main_button.connect("clicked", self._on_main_button_clicked)
        self.append(self.main_button)

        # 2. The dropdown button for the menu
        popover = self._build_popover()
        self.menu_button = Gtk.MenuButton(
            icon_name="pan-down-symbolic",
            popover=popover,
            tooltip_text=_("Show all options"),
        )
        self.append(self.menu_button)

        # Set the initial state of the main button
        self._set_active_action(self._last_action_index, fire_action=False)

    def set_sensitive(self, sensitive: bool):
        """Sets the sensitivity of the entire composite button."""
        self.main_button.set_sensitive(sensitive)
        self.menu_button.set_sensitive(sensitive)

    def _build_popover(self) -> Gtk.Popover:
        """Creates the popover menu with buttons for all actions."""
        popover = Gtk.Popover()
        list_box = Gtk.ListBox()
        list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        list_box.get_style_context().add_class("popover-list")
        popover.set_child(list_box)

        for i, (name, icon_name, _) in enumerate(self.actions):
            row = Gtk.ListBoxRow()
            button = Gtk.Button()
            button.set_has_frame(False)
            content = Gtk.Box(spacing=6)
            content.append(get_icon(icon_name))
            content.append(Gtk.Label(label=name))
            button.set_child(content)
            # Use a lambda that captures the current index `i`
            button.connect(
                "clicked", lambda _, idx=i: self._on_menu_item_clicked(idx)
            )
            row.set_child(button)
            list_box.append(row)

        return popover

    def _on_main_button_clicked(self, _):
        """Fires the last-used action."""
        _, _, signal_obj = self.actions[self._last_action_index]
        signal_obj.send(self)

    def _on_menu_item_clicked(self, index: int):
        """Called when a user clicks an item in the popover menu."""
        self.menu_button.get_popover().popdown()
        self._set_active_action(index, fire_action=True)

    def _set_active_action(self, index: int, fire_action: bool):
        """
        Updates the main button to reflect the new active action and
        optionally fires its signal.

        Args:
            index: The index of the action to set as active.
            fire_action: If True, the action's signal is sent.
        """
        self._last_action_index = index
        name, icon_name, signal_obj = self.actions[index]
        self.main_button.set_child(get_icon(icon_name))
        self.main_button.set_tooltip_text(name)

        if fire_action:
            signal_obj.send(self)
