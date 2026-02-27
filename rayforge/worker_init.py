import builtins
import logging
import os
import sys
from pathlib import Path


def initialize_worker():
    """
    Sets up the minimal environment required for a worker subprocess.

    This function is lightweight and has no dangerous imports. It is the
    designated `worker_initializer` for the TaskManager.
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

    # Import builtin packages to register their producers.
    # This must happen before any from_dict() deserialization.
    try:
        from rayforge.builtin_packages import ensure_loaded

        ensure_loaded()
    except ImportError:
        pass

    logging.getLogger(__name__).debug(
        "Worker process initialized successfully."
    )
