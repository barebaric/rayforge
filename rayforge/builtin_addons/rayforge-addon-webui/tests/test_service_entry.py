"""Tests for the webui addon's service entry point."""

from rayforge.addon_mgr.addon import Addon
from rayforge.config import BUILTIN_ADDONS_DIR
from rayforge.shared.util.versioning import UnknownVersion

SERVICE_KEY = "webui"


class TestServiceEntryPoint:
    def test_manifest_declares_service_entry_point(self):
        addon = Addon.load_from_directory(
            BUILTIN_ADDONS_DIR / "rayforge-addon-webui",
            version=UnknownVersion,
        )
        assert addon.metadata.provides.service == "webui.service"
        assert addon.validate()

    def test_service_loads_in_headless_mode(self, webui_registry):
        factory = webui_registry.get(SERVICE_KEY)
        assert callable(factory)

    def test_service_not_loaded_in_gui_mode(self, gui_registry):
        assert gui_registry.get(SERVICE_KEY) is None
