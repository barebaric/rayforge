from typing import Dict, List, Optional, Type, Callable, TYPE_CHECKING

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

    def register(self, step_class: Type["Step"]) -> None:
        """
        Register a step class.

        Args:
            step_class: The Step subclass to register.
                        The class name is used as the registry key.
        """
        name = step_class.__name__
        self._steps[name] = step_class

    def get(self, name: str) -> Optional[Type["Step"]]:
        """
        Look up a step class by name.

        Args:
            name: The class name of the step.

        Returns:
            The step class, or None if not found.
        """
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
