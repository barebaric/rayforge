"""Helpers for detecting the platform and the way Rayforge is packaged.

Platform and packaging questions are asked in several places, and the
answers decide which update path is offered to the user. Keeping them
here avoids scattering the same environment checks across modules.
"""

import ctypes
import sys

_APPMODEL_ERROR_NO_PACKAGE = 15700
_ERROR_INSUFFICIENT_BUFFER = 122


def is_windows() -> bool:
    """Returns whether Rayforge is running on Windows."""
    return sys.platform == "win32"


def is_frozen() -> bool:
    """Returns whether Rayforge runs from a PyInstaller bundle."""
    return hasattr(sys, "_MEIPASS")


def is_msix() -> bool:
    """Returns whether Rayforge runs with MSIX package identity."""
    if not is_windows():
        return False
    kernel32 = getattr(getattr(ctypes, "windll", None), "kernel32", None)
    if kernel32 is None:
        return False
    try:
        result = kernel32.GetCurrentPackageFullName(
            ctypes.byref(ctypes.c_uint32()), None
        )
    except (AttributeError, OSError):
        return False
    return result == _ERROR_INSUFFICIENT_BUFFER


def supports_self_update() -> bool:
    """Returns whether Rayforge can install an update on its own.

    Only the Windows bundle built by our release workflow ships with an
    installer that knows how to replace a running installation, so only
    that combination may download and launch an installer. MSIX packages
    are updated by the Store or App Installer instead.
    """
    return is_windows() and is_frozen() and not is_msix()
