import builtins
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from rayforge.addon_mgr.lazy_loader import (
    ensure_addon_namespaces,
    install_addon_finder,
)


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

    from rayforge.context import get_context
    from rayforge.core.step_registry import step_registry
    from rayforge.doceditor.layout.registry import layout_registry
    from rayforge.pipeline.producer.registry import producer_registry
    from rayforge.pipeline.transformer.registry import transformer_registry

    logger = logging.getLogger(__name__)

    context = get_context()
    context._headless = True

    manifest = (
        _shared_state_cache.get("addon_manifest")
        if _shared_state_cache
        else None
    )

    if not manifest:
        logger.warning("Cannot load worker addons: manifest not available.")
        return

    if _shared_state_cache:
        install_addon_finder(shared_dict=_shared_state_cache)
    ensure_addon_namespaces(manifest)

    for module_name in manifest.enabled_backend_modules:
        try:
            module = importlib.import_module(module_name)
            context.plugin_mgr.register(module)
        except Exception as e:
            logger.error(
                f"Failed to load enabled backend module '{module_name}': {e}"
            )

    context.plugin_mgr.hook.register_steps(step_registry=step_registry)
    context.plugin_mgr.hook.register_producers(
        producer_registry=producer_registry
    )
    context.plugin_mgr.hook.register_transformers(
        transformer_registry=transformer_registry
    )
    context.plugin_mgr.hook.register_layout_strategies(
        layout_registry=layout_registry
    )

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
    elif hasattr(sys, "_MEIPASS") and sys.platform == "win32":
        # Windows PyInstaller bundles need explicit DLL search path
        # for spawned subprocesses to find cairo, rsvg, etc.
        base_dir = Path(sys._MEIPASS)
        try:
            os.add_dll_directory(str(base_dir))
        except OSError:
            pass

    logger = logging.getLogger(__name__)
    if shared_state is None:
        logger.error("Worker cannot initialize without shared_state.")
        return

    manifest = shared_state.get("addon_manifest")
    if not manifest:
        logger.warning(
            "Addon manifest not yet available in shared state. "
            "Addon finder will be installed lazily."
        )
    else:
        install_addon_finder(shared_dict=shared_state)
        ensure_addon_namespaces(manifest)

    logger.debug("Worker process initialized.")
