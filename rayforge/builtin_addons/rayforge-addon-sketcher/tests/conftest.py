"""
Pytest configuration for sketcher builtin addon tests.

This conftest ensures that the sketcher module is importable
by adding the addon directory to sys.path before tests run.
"""

import sys
import pytest
from pathlib import Path

# Add the addon directory to sys.path so that the sketcher module
# can be imported from tests
addon_dir = Path(__file__).parent.parent
sys.path.insert(0, str(addon_dir))


@pytest.fixture(scope="session", autouse=True)
def register_sketch_asset_type():
    """Register Sketch asset type for tests that need serialization."""
    from rayforge.core.asset_registry import asset_type_registry
    from sketcher.core import Sketch

    asset_type_registry.register(Sketch, "sketch", "sketcher")
    yield
    asset_type_registry.unregister("sketch")
