import builtins
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def initialize_worker(shared_state=None):
    """
    Sets up minimal environment required for a worker subprocess.

    Installs a fallback gettext translator and platform-specific
    dynamic library paths for PyInstaller bundles.
    """
    # Install a fallback gettext translator. This ensures the '_'
    # function exists during the module import phase.
    if not hasattr(builtins, "_"):
        builtins._ = lambda s: s  # type: ignore[attr-defined]

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
    logger.debug("Worker process initialized.")
