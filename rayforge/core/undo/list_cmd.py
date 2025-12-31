from typing import Any, List, Optional, Callable
from .command import Command


class ListItemCommand(Command):
    """A command for adding or removing an item from a list-like container."""

    def __init__(
        self,
        owner_obj: Any,
        item: Any,
        undo_command: str,
        redo_command: str,
        on_change_callback: Optional[Callable[[], None]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name, on_change_callback)
        self.owner_obj = owner_obj
        self.item = item
        self.redo_command = getattr(owner_obj, redo_command)
        self.undo_command = getattr(owner_obj, undo_command)

    def execute(self) -> None:
        """Executes the redo action."""
        self.redo_command(self.item)
        if self.on_change_callback:
            self.on_change_callback()

    def undo(self) -> None:
        """Executes the undo action."""
        self.undo_command(self.item)
        if self.on_change_callback:
            self.on_change_callback()


class ReorderListCommand(Command):
    """A command to handle the reordering of a list."""

    def __init__(
        self,
        target_obj: Any,
        list_property_name: str,
        new_list: List[Any],
        setter_method_name: Optional[str] = None,
        on_change_callback: Optional[Callable[[], None]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name, on_change_callback)
        self.target_obj = target_obj
        self.list_property_name = list_property_name
        self.new_list = list(new_list)
        self.setter_method_name = setter_method_name
        self.old_list = list(getattr(target_obj, list_property_name))

    def _set_list(self, new_order: List[Any]):
        if self.setter_method_name:
            setter_func = getattr(self.target_obj, self.setter_method_name)
            setter_func(new_order)
        else:
            setattr(self.target_obj, self.list_property_name, new_order)

    def execute(self) -> None:
        """Applies the new order to the list."""
        self._set_list(self.new_list)
        if self.on_change_callback:
            self.on_change_callback()

    def undo(self) -> None:
        """Restores the original order of the list."""
        self._set_list(self.old_list)
        if self.on_change_callback:
            self.on_change_callback()
