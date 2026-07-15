"""
Registry for assembler functions.

Provides a name-based lookup for assembler callables, allowing steps
to delegate to raygeo assemblers without importing raygeo directly.
"""

import logging
from typing import Any, Callable, Dict, Optional, Set

from rayforge.worker_init import ensure_addons_loaded

logger = logging.getLogger(__name__)


class AssemblerRegistry:
    """
    Registry for assembler functions.

    Maps string names to assembler callables so that steps can look
    up and call assemblers without importing raygeo directly.
    """

    def __init__(self):
        self._assemblers: Dict[str, Callable[..., Any]] = {}
        self._addon_items: Dict[str, Set[str]] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        addon_name: Optional[str] = None,
    ) -> None:
        """
        Register an assembler.

        Args:
            name: The name to register the assembler under.
            func: The assembler callable.
            addon_name: Optional addon name for cleanup tracking.
        """
        if name in self._assemblers:
            logger.warning(
                "Assembler '%s' already registered, overwriting", name
            )
        self._assemblers[name] = func
        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(name)

    def unregister(self, name: str) -> bool:
        """
        Unregister an assembler by name.

        Returns:
            True if the assembler was unregistered, False if not found.
        """
        if name in self._assemblers:
            del self._assemblers[name]
            for addon_name, items in self._addon_items.items():
                items.discard(name)
            return True
        return False

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all assemblers registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of assemblers unregistered.
        """
        if addon_name not in self._addon_items:
            return 0
        items = self._addon_items.pop(addon_name)
        count = 0
        for name in items:
            if name in self._assemblers:
                del self._assemblers[name]
                count += 1
        return count

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        """
        Look up an assembler by name.

        Returns:
            The assembler callable, or None if not found.
        """
        func = self._assemblers.get(name)
        if func is None:
            ensure_addons_loaded()
            func = self._assemblers.get(name)
        return func

    def assemble(
        self, name: str, part: Any = None, **kwargs: Any
    ) -> Any:
        """
        Look up an assembler by name and call it.

        Args:
            name: The registered name of the assembler.
            part: The Part to pass as the first positional argument,
                  or None for assemblers that don't need one.
            **kwargs: Parameters to forward to the assembler function.

        Returns:
            The result of calling the assembler function.

        Raises:
            KeyError: If no assembler is registered under the given name.
        """
        func = self._assemblers.get(name)
        if func is None:
            ensure_addons_loaded()
            func = self._assemblers[name]
        if part is not None:
            return func(part, **kwargs)
        return func(**kwargs)

    def all_assemblers(self) -> Dict[str, Callable[..., Any]]:
        """Return a copy of all registered assemblers."""
        return self._assemblers.copy()


assembler_registry = AssemblerRegistry()
