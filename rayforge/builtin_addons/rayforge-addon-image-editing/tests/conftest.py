"""
Pytest configuration for image_editing builtin addon tests.
"""

import pytest
from image_editing.ops import ALL_OPS

from rayforge.core.image_edit import image_edit_registry

ADDON_NAME = "image_editing"


@pytest.fixture(scope="session", autouse=True)
def register_image_edits():
    """Registers the addon's image edit ops for all tests."""
    for op_class in ALL_OPS:
        image_edit_registry.register(op_class, addon_name=ADDON_NAME)
    yield
    image_edit_registry.unregister_all_from_addon(ADDON_NAME)
