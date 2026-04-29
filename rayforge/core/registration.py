import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegistryEntry:
    hook_name: str
    param_name: str
    module_path: str
    attr_name: str
    worker_ok: bool
    needs_window: bool


REGISTRY_TABLE = [
    RegistryEntry(
        "register_steps",
        "step_registry",
        "rayforge.core.step_registry",
        "step_registry",
        worker_ok=True,
        needs_window=False,
    ),
    RegistryEntry(
        "register_producers",
        "producer_registry",
        "rayforge.pipeline.producer.registry",
        "producer_registry",
        worker_ok=True,
        needs_window=False,
    ),
    RegistryEntry(
        "register_transformers",
        "transformer_registry",
        "rayforge.pipeline.transformer.registry",
        "transformer_registry",
        worker_ok=True,
        needs_window=False,
    ),
    RegistryEntry(
        "register_layout_strategies",
        "layout_registry",
        "rayforge.doceditor.layout.registry",
        "layout_registry",
        worker_ok=True,
        needs_window=False,
    ),
    RegistryEntry(
        "register_asset_types",
        "asset_type_registry",
        "rayforge.core.asset_registry",
        "asset_type_registry",
        worker_ok=True,
        needs_window=False,
    ),
    RegistryEntry(
        "register_renderers",
        "renderer_registry",
        "rayforge.image",
        "renderer_registry",
        worker_ok=True,
        needs_window=False,
    ),
    RegistryEntry(
        "register_commands",
        "command_registry",
        "rayforge.doceditor.command_registry",
        "command_registry",
        worker_ok=False,
        needs_window=False,
    ),
    RegistryEntry(
        "register_exporters",
        "exporter_registry",
        "rayforge.image",
        "exporter_registry",
        worker_ok=False,
        needs_window=False,
    ),
    RegistryEntry(
        "register_importers",
        "importer_registry",
        "rayforge.image",
        "importer_registry",
        worker_ok=False,
        needs_window=False,
    ),
    RegistryEntry(
        "register_actions",
        "action_registry",
        "rayforge.ui_gtk.action_registry",
        "action_registry",
        worker_ok=False,
        needs_window=True,
    ),
]

LAZY_MANAGERS = {
    "library_manager": (
        "register_material_libraries",
        "library_manager",
    ),
    "model_manager": (
        "register_model_libraries",
        "model_manager",
    ),
}


def _import_registry(entry: RegistryEntry) -> Any:
    module = importlib.import_module(entry.module_path)
    return getattr(module, entry.attr_name)


def get_registries(headless: bool = False) -> Dict[str, Any]:
    """
    Import and return a dict of all active registries.

    The returned dict maps param_name -> registry instance for all
    registries appropriate for the given mode.
    """
    result: Dict[str, Any] = {}
    for entry in REGISTRY_TABLE:
        if headless and not entry.worker_ok:
            continue
        result[entry.param_name] = _import_registry(entry)
    if not headless:
        from rayforge.ui_gtk.actions import action_extension_registry
        from rayforge.ui_gtk.canvas2d.context_menu import (
            context_menu_extension_registry,
        )
        from rayforge.ui_gtk.doceditor.property_providers import (
            property_provider_registry,
        )

        result["action_extension_registry"] = action_extension_registry
        result["context_menu_extension_registry"] = (
            context_menu_extension_registry
        )
        result["property_provider_registry"] = property_provider_registry
    return result


def call_registration_hooks(
    plugin_mgr,
    headless: bool = False,
    registries: Optional[Dict[str, Any]] = None,
    window_required: bool = False,
):
    """
    Call all appropriate registration hooks on the plugin manager.

    This is the single entry point for registration hook invocation,
    used during app startup, addon enable/reload, and worker init.

    Args:
        plugin_mgr: The pluggy PluginManager instance.
        headless: If True, skip GUI-only registries (worker mode).
        registries: Optional dict of pre-loaded registries. Entries not
            found here will be imported from their module_path.
        window_required: If True, call only hooks that register
            actions and other UI elements that depend on the main
            window being available (e.g. during addon enable/reload
            at runtime). If False, call all other hooks (startup and
            worker init).
    """
    registries = registries or {}
    for entry in REGISTRY_TABLE:
        if window_required and not entry.needs_window:
            continue
        if not window_required and entry.needs_window:
            continue
        if headless and not entry.worker_ok:
            continue
        registry = registries.get(entry.param_name)
        if registry is None:
            try:
                registry = _import_registry(entry)
            except (ImportError, AttributeError):
                continue
        getattr(plugin_mgr.hook, entry.hook_name)(
            **{entry.param_name: registry}
        )
    if not window_required:
        for key, (hook_name, param_name) in LAZY_MANAGERS.items():
            registry = registries.get(key)
            if registry is not None:
                logger.debug(f"Calling {hook_name} hook")
                getattr(plugin_mgr.hook, hook_name)(
                    **{param_name: registry}
                )
