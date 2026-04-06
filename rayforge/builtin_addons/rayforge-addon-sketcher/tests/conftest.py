"""
Pytest configuration for sketcher builtin addon tests.

This conftest ensures that the sketcher module is importable
by adding the addon directory to sys.path before tests run.
"""

import sys
import pytest
from pathlib import Path

_gtk_available = True
try:
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    import gi.repository.Adw  # noqa: F401, check availability
except (ValueError, ImportError):
    _gtk_available = False


# Add the addon directory to sys.path so that the sketcher module
# can be imported from tests
addon_dir = Path(__file__).parent.parent
sys.path.insert(0, str(addon_dir))


def pytest_ignore_collect(collection_path, config):
    """Skip UI test files when GTK/Adw is not available."""
    if not _gtk_available:
        path_str = str(collection_path)
        if "/ui_gtk/" in path_str:
            return True
    return False


@pytest.fixture(scope="session", autouse=True)
def register_sketch_asset_type():
    """Register Sketch asset type for tests that need serialization."""
    from rayforge.core.asset_registry import asset_type_registry
    from sketcher.core import Sketch

    asset_type_registry.register(Sketch, "sketch", "sketcher")
    yield
    asset_type_registry.unregister("sketch")
