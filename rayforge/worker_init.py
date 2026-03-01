import builtins
import logging
import os
import sys
from pathlib import Path


_worker_addons_loaded = False


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

    import pluggy
    from rayforge.config import (
        BUILTIN_PACKAGES_DIR,
        PACKAGES_DIR,
        CONFIG_DIR,
    )
    from rayforge.core.hooks import RayforgeSpecs
    from rayforge.core.step_registry import step_registry
    from rayforge.pipeline.producer.registry import producer_registry
    from rayforge.package_mgr.package_manager import PackageManager
    from rayforge.core.addon_config import AddonConfig

    logger = logging.getLogger(__name__)

    plugin_mgr = pluggy.PluginManager("rayforge")
    plugin_mgr.add_hookspecs(RayforgeSpecs)

    # Load the actual addon config to respect enabled/disabled state
    addon_config = AddonConfig(CONFIG_DIR)
    addon_config.load()

    package_mgr = PackageManager(
        [BUILTIN_PACKAGES_DIR, PACKAGES_DIR],
        PACKAGES_DIR,
        plugin_mgr,
        addon_config,
    )
    package_mgr.set_registries(
        {
            "step_registry": step_registry,
            "producer_registry": producer_registry,
        }
    )
    # backend_only=True skips frontend/widgets to avoid pulling in GTK
    package_mgr.load_installed_packages(backend_only=True)

    # Only register backend hooks, not frontend (widgets, menus, actions)
    plugin_mgr.hook.register_steps(step_registry=step_registry)
    plugin_mgr.hook.register_producers(producer_registry=producer_registry)

    logger.debug("Worker addons loaded.")


def invalidate_worker_addons_cache():
    """
    Invalidate the addon cache so subsequent calls to ensure_addons_loaded()
    will reload addons from disk.

    This should be called when addons are installed or removed at runtime,
    followed by restarting the worker pool to pick up the changes.
    """
    global _worker_addons_loaded
    _worker_addons_loaded = False


def initialize_worker():
    """
    Sets up the minimal environment required for a worker subprocess.

    This function is lightweight and has no dangerous imports. It is the
    designated `worker_initializer` for the TaskManager.

    Addons are loaded lazily via ensure_addons_loaded() when first needed
    for deserializing producers, rather than at worker startup.
    """
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

    logging.getLogger(__name__).debug("Worker process initialized.")
