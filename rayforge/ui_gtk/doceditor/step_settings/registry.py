from typing import Any, Dict, Optional, Set, Type


class StepWidgetRegistry:
    """
    Registry for step settings widgets.

    Allows explicit registration of widget classes for specific
    producer or transformer types.
    """

    def __init__(self):
        self._widgets: Dict[str, Type[Any]] = {}
        self._addon_items: Dict[str, Set[str]] = {}

    def register(
        self,
        component_class: Type[Any],
        widget_class: Type[Any],
        addon_name: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Register a widget class for a component type.

        Args:
            component_class: The producer or transformer class.
            widget_class: The widget class to use for this component.
            addon_name: Optional name of the addon registering this widget.
                        Used for cleanup when addon is unloaded.
            name: Optional name override. If not provided, uses
                  component_class.__name__.
        """
        component_name = name if name else component_class.__name__
        self._widgets[component_name] = widget_class
        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(component_name)

    def unregister(self, component_name: str) -> bool:
        """
        Unregister a widget class by component name.

        Args:
            component_name: The name of the producer or transformer type.

        Returns:
            True if the widget was unregistered, False if not found.
        """
        if component_name in self._widgets:
            del self._widgets[component_name]
            for addon_name, items in self._addon_items.items():
                items.discard(component_name)
            return True
        return False

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all widgets registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of widgets unregistered.
        """
        if addon_name not in self._addon_items:
            return 0
        items = self._addon_items.pop(addon_name)
        count = 0
        for name in items:
            if name in self._widgets:
                del self._widgets[name]
                count += 1
        return count

    def get(self, component_name: str) -> Optional[Type[Any]]:
        """
        Look up a widget class by component name.

        Args:
            component_name: The name of the producer or transformer type.

        Returns:
            The widget class, or None if not found.
        """
        return self._widgets.get(component_name)

    def all_widgets(self) -> Dict[str, Type[Any]]:
        """
        Return a copy of all registered widgets.

        Returns:
            Dictionary mapping component names to widget classes.
        """
        return self._widgets.copy()


step_widget_registry = StepWidgetRegistry()
