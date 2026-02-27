from typing import Any, Dict, Optional, Type


class StepWidgetRegistry:
    """
    Registry for step settings widgets.

    Allows explicit registration of widget classes for specific
    producer or transformer type names.
    """

    def __init__(self):
        self._widgets: Dict[str, Type[Any]] = {}

    def register(self, component_name: str, widget_class: Type[Any]) -> None:
        """
        Register a widget class for a component type name.

        Args:
            component_name: The name of the producer or transformer type.
            widget_class: The widget class to use for this component.
        """
        self._widgets[component_name] = widget_class

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
