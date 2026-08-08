"""
tool_library addon: cutting-tool library for CNC machining.

Worker entry point. Publishes the :func:`get_tool_manager` accessor as
the ``"tool_manager"`` service, which other addons resolve via the
global service registry::

    from rayforge.core.service_registry import service_registry

    get_tool_manager = service_registry.get("tool_manager")
    if get_tool_manager is not None:
        tool = get_tool_manager().get(tool_uid)
"""

import logging
from typing import Optional

from rayforge.config import CONFIG_DIR
from rayforge.core.hooks import hookimpl
from rayforge.core.service_registry import ServiceRegistry

from .manager import ToolManager

logger = logging.getLogger(__name__)

_manager: ToolManager | None = None


def get_tool_manager() -> ToolManager:
    """Return the process-wide :class:`ToolManager` singleton (lazy)."""
    global _manager
    if _manager is None:
        _manager = ToolManager(CONFIG_DIR / "tools")
    return _manager


@hookimpl
def register_services(service_registry: ServiceRegistry) -> None:
    service_registry.register(
        "tool_manager", get_tool_manager, addon_name="tool_library"
    )
