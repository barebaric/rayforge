import os
import sys
from pathlib import Path

if hasattr(sys, "_MEIPASS"):
    frameworks_dir = Path(sys._MEIPASS).parent / "Frameworks"
    lib_path = str(frameworks_dir)
    existing_dyld = os.environ.get("DYLD_LIBRARY_PATH")
    os.environ["DYLD_LIBRARY_PATH"] = (
        lib_path if not existing_dyld else f"{lib_path}:{existing_dyld}"
    )
    os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", lib_path)

    typelib_dir = frameworks_dir / "gi_typelibs"
    if typelib_dir.exists():
        os.environ["GI_TYPELIB_PATH"] = str(typelib_dir)

    gio_modules = frameworks_dir / "gio_modules"
    if gio_modules.exists():
        os.environ.setdefault("GIO_EXTRA_MODULES", str(gio_modules))
