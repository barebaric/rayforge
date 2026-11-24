"""Helpers for detecting the platform and the way Rayforge is packaged.

Platform and packaging questions are asked in several places, and the
answers decide which update path is offered to the user. Keeping them
here avoids scattering the same environment checks across modules.
"""

import sys


def is_windows() -> bool:
    """Returns whether Rayforge is running on Windows."""
    return sys.platform == "win32"


def is_frozen() -> bool:
    """Returns whether Rayforge runs from a PyInstaller bundle."""
    return hasattr(sys, "_MEIPASS")


def supports_self_update() -> bool:
    """Returns whether Rayforge can install an update on its own.

    Only the Windows bundle built by our release workflow ships with an
    installer that knows how to replace a running installation, so only
    that combination may download and launch an installer.
    """
    return is_windows() and is_frozen()
