import builtins
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import pluggy
from rayforge.addon_mgr.lazy_loader import (
    ensure_addon_namespaces,
    install_addon_finder,
)
from rayforge.core.hooks import RayforgeSpecs


_worker_addons_loaded = False
_shared_state_cache: Optional[Dict] = None


def ensure_addons_loaded():
    """
    Lazily load addons to populate producer and step registries.

    This is called on first use rather than at worker startup to avoid
    slowing down worker processes that may not need addon functionality.
    """
    global _worker_addons_loaded

    if _worker_addons_loaded:
        return

    _worker_addons_loaded = True

    from rayforge.core.step_registry import step_registry
    from rayforge.doceditor.layout.registry import layout_registry
    from rayforge.pipeline.producer.registry import producer_registry

    logger = logging.getLogger(__name__)

    plugin_mgr = pluggy.PluginManager("rayforge")
    plugin_mgr.add_hookspecs(RayforgeSpecs)

    manifest = None

    # Priority 1: Use cached state (Worker process scenario)
    if _shared_state_cache is not None:
        manifest = _shared_state_cache.get("addon_manifest")
    else:
        # Priority 2: Use TaskManager (Main process scenario)
        # We must be careful not to trigger TaskManager creation in a worker
        try:
            from rayforge.shared.tasker import task_mgr

            # accessing get_shared_state on the proxy triggers initialization
            # which is safe in the main process but fatal in a worker.
            # We assume that if _shared_state_cache is None, we are either
            # in the main process or something went wrong with initialization.
            shared_state = task_mgr.get_shared_state()
            manifest = shared_state.get("addon_manifest")
        except Exception as e:
            # Catches AssertionError (daemonic process) or other init errors
            logger.debug(f"Could not retrieve manifest from task_mgr: {e}")

    if not manifest:
        logger.warning("Cannot load worker addons: manifest not available.")
        return

    for module_name in manifest.enabled_backend_modules:
        try:
            module = importlib.import_module(module_name)
            plugin_mgr.register(module)
        except Exception as e:
            logger.error(
                f"Failed to load enabled backend module '{module_name}': {e}"
            )

    # Only register backend hooks, not frontend (widgets, menus, actions)
    plugin_mgr.hook.register_steps(step_registry=step_registry)
    plugin_mgr.hook.register_producers(producer_registry=producer_registry)
    plugin_mgr.hook.register_layout_strategies(layout_registry=layout_registry)

    logger.debug("Worker addons loaded from manifest.")


def invalidate_worker_addons_cache():
    """
    Invalidate addon cache so subsequent calls to ensure_addons_loaded()
    will reload addons.

    This should be called when addons are installed or removed at runtime,
    followed by restarting the worker pool to pick up the changes.
    """
    global _worker_addons_loaded
    _worker_addons_loaded = False


def initialize_worker(shared_state: Optional[Dict] = None):
    """
    Sets up minimal environment required for a worker subprocess.

    This function is lightweight and has no dangerous imports. It is
    designated `worker_initializer` for TaskManager.

    Addons are loaded lazily via ensure_addons_loaded() when first needed
    for deserializing producers, rather than at worker startup.

    Args:
        shared_state: Shared dict for worker initialization data.
            Contains the 'addon_manifest' key with the pre-computed structure.
    """
    global _shared_state_cache
    _shared_state_cache = shared_state

    # Install a fallback gettext translator. This ensures the '_'
    # function exists during the module import phase.
    if not hasattr(builtins, "_"):
        setattr(builtins, "_", lambda s: s)

    if hasattr(sys, "_MEIPASS") and sys.platform == "darwin":
        # macOS PyInstaller bundles require specific environment variables
        # for dynamic linking and GObject Introspection to work correctly
        # in worker subprocesses.
        frameworks_dir = Path(sys._MEIPASS).parent / "Frameworks"
        lib_path = str(frameworks_dir)
        # DYLD_LIBRARY_PATH: Directories for dynamic linker to search
        existing_dyld = os.environ.get("DYLD_LIBRARY_PATH")
        os.environ["DYLD_LIBRARY_PATH"] = (
            lib_path if not existing_dyld else f"{lib_path}:{existing_dyld}"
        )
        # DYLD_FALLBACK_LIBRARY_PATH: Fallback if DYLD_LIBRARY_PATH fails
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", lib_path)
        # GI_TYPELIB_PATH: Path to GObject Introspection typelib files
        bundled_typelibs = frameworks_dir / "gi_typelibs"
        if bundled_typelibs.exists():
            os.environ["GI_TYPELIB_PATH"] = str(bundled_typelibs.resolve())
        # GIO_EXTRA_MODULES: Path to additional GIO modules
        bundled_gio_modules = frameworks_dir / "gio_modules"
        if bundled_gio_modules.exists():
            os.environ.setdefault(
                "GIO_EXTRA_MODULES", str(bundled_gio_modules)
            )

    logger = logging.getLogger(__name__)
    if not shared_state:
        logger.error("Worker cannot initialize without shared_state.")
        return

    manifest = shared_state.get("addon_manifest")
    if not manifest:
        logger.warning("Addon manifest not found in shared state.")
    else:
        install_addon_finder(shared_dict=shared_state)
        ensure_addon_namespaces(manifest)

    logger.debug("Worker process initialized with addon manifest.")
