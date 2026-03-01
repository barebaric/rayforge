from typing import Dict, List, Optional, Set, Type, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .step import Step


class StepRegistry:
    """
    Registry for Step classes.

    Allows explicit registration of step types for lookup by name.
    Supports polymorphic deserialization and provides access to
    step factory methods for UI menus.
    """

    def __init__(self):
        self._steps: Dict[str, Type["Step"]] = {}
        self._addon_items: Dict[str, Set[str]] = {}

    def register(
        self, step_class: Type["Step"], addon_name: Optional[str] = None
    ) -> None:
        """
        Register a step class.

        Args:
            step_class: The Step subclass to register.
                        The class name is used as the registry key.
            addon_name: Optional name of the addon registering this step.
                        Used for cleanup when addon is unloaded.
        """
        name = step_class.__name__
        self._steps[name] = step_class
        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(name)

    def unregister(self, name: str) -> bool:
        """
        Unregister a step class by name.

        Args:
            name: The class name of the step to unregister.

        Returns:
            True if the step was unregistered, False if not found.
        """
        if name in self._steps:
            del self._steps[name]
            for addon_name, items in self._addon_items.items():
                items.discard(name)
            return True
        return False

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all steps registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of steps unregistered.
        """
        if addon_name not in self._addon_items:
            return 0
        items = self._addon_items.pop(addon_name)
        count = 0
        for name in items:
            if name in self._steps:
                del self._steps[name]
                count += 1
        return count

    def get(self, name: str) -> Optional[Type["Step"]]:
        """
        Look up a step class by name.

        Args:
            name: The class name of the step.

        Returns:
            The step class, or None if not found.
        """
        # Ensure addons are loaded in worker processes so the registry
        # is populated before we try to look up step classes.
        from rayforge.worker_init import ensure_addons_loaded

        ensure_addons_loaded()

        return self._steps.get(name)

    def get_factories(self) -> List[Callable]:
        """
        Return all registered step factory methods.

        Returns:
            List of callable `create` class methods from registered
            step classes, excluding hidden steps.
        """
        factories: List[Callable] = []
        for cls in self._steps.values():
            if cls.HIDDEN:
                continue
            factories.append(cls.create)
        return factories

    def all_steps(self) -> Dict[str, Type["Step"]]:
        """
        Return a copy of all registered steps.

        Returns:
            Dictionary mapping step names to classes.
        """
        return self._steps.copy()


step_registry = StepRegistry()
