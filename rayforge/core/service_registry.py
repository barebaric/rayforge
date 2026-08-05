"""
Generic service registry for addon-provided services.

Addons publish services (callables, instances, classes) under a string
key via the ``register_services`` hook; consumers look them up by key.
This lets one addon expose functionality to another without a direct
cross-package import.

Implements the :class:`~rayforge.addon_mgr.addon_manager.AddonRegistry`
protocol so a service is removed automatically when its addon unloads.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Maps a string key to a service, tracking the owning addon."""

    def __init__(self) -> None:
        self._services: Dict[str, Tuple[Any, str]] = {}

    def register(self, key: str, service: Any, addon_name: str = "") -> None:
        """
        Register (or replace) a service under ``key``.

        Args:
            key: Lookup key, by convention a lowercase noun
                (``"tool_manager"``).
            service: The service (commonly a zero-arg accessor callable).
            addon_name: Canonical name of the contributing addon, used
                for cleanup on unload.
        """
        self._services[key] = (service, addon_name)
        logger.debug(f"Registered service '{key}' for '{addon_name}'")

    def get(self, key: str) -> Optional[Any]:
        """Return the service registered under ``key``, or ``None``."""
        entry = self._services.get(key)
        return entry[0] if entry is not None else None

    def keys(self) -> List[str]:
        """Return all registered service keys."""
        return list(self._services.keys())

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """Remove every service registered by the named addon."""
        before = len(self._services)
        self._services = {
            k: v for k, v in self._services.items() if v[1] != addon_name
        }
        removed = before - len(self._services)
        if removed:
            logger.info(f"Removed {removed} services from '{addon_name}'")
        return removed


service_registry = ServiceRegistry()
