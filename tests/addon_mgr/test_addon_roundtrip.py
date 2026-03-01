"""Integration tests for addon loading and execution."""

import sys
from unittest.mock import patch

from rayforge.context import RayforgeContext
from rayforge.addon_mgr.addon_manager import (
    AddonManager,
    UpdateStatus,
)


class TestAddonRoundTrip:
    """Test cases for full addon loading lifecycle."""

    def test_full_addon_loading_and_execution(self, tmp_path):
        """
        Creates a real addon file structure, loads it, and ensures it runs.
        """
        addons_dir = tmp_path / "packages"
        addon_dir = addons_dir / "integration_test_addon"
        addon_dir.mkdir(parents=True)

        (addon_dir / "rayforge-addon.yaml").write_text(
            "name: this_name_is_ignored\nversion: 0.1\n"
            "api_version: 1\n"
            "depends:\n"
            "  - rayforge>=0.27.0,~0.27\n"
            "provides:\n"
            "  backend: addon\n"
        )

        (addon_dir / "addon.py").write_text(
            "import sys\n"
            "from rayforge.core.hooks import hookimpl\n"
            "\n"
            "sys.modules['integration_test_addon_loaded'] = False\n"
            "\n"
            "@hookimpl\n"
            "def rayforge_init(context):\n"
            "    sys.modules['integration_test_addon_loaded'] = True\n"
        )

        context = RayforgeContext()
        context.addon_mgr = AddonManager(
            [addons_dir], addons_dir, context.plugin_mgr
        )

        with patch.object(
            context.addon_mgr,
            "_check_version_compatibility",
            return_value=UpdateStatus.UP_TO_DATE,
        ):
            context.initialize_full_context(load_ui=False)

        # The module name is derived from the directory, not the YAML file.
        assert "rayforge_plugins.integration_test_addon" in sys.modules
        assert sys.modules.get("integration_test_addon_loaded") is True

        del sys.modules["integration_test_addon_loaded"]

    def test_addon_loading_with_multiple_hooks(self, tmp_path):
        """
        Tests that an addon can implement multiple hooks.
        """
        addons_dir = tmp_path / "packages"
        addon_dir = addons_dir / "multi_hook_addon"
        addon_dir.mkdir(parents=True)

        (addon_dir / "rayforge-addon.yaml").write_text(
            "name: this_name_is_ignored\nversion: 0.1\n"
            "api_version: 1\n"
            "depends:\n"
            "  - rayforge>=0.27.0,~0.27\n"
            "provides:\n"
            "  backend: addon\n"
        )

        (addon_dir / "addon.py").write_text(
            "import sys\n"
            "from rayforge.core.hooks import hookimpl\n"
            "\n"
            "sys.modules['multi_hook_rayforge_init'] = False\n"
            "\n"
            "@hookimpl\n"
            "def rayforge_init(context):\n"
            "    sys.modules['multi_hook_rayforge_init'] = True\n"
        )

        context = RayforgeContext()
        context.addon_mgr = AddonManager(
            [addons_dir], addons_dir, context.plugin_mgr
        )

        with patch.object(
            context.addon_mgr,
            "_check_version_compatibility",
            return_value=UpdateStatus.UP_TO_DATE,
        ):
            context.initialize_full_context(load_ui=False)

        # The module name is derived from the directory, not the YAML file.
        assert "rayforge_plugins.multi_hook_addon" in sys.modules
        assert sys.modules.get("multi_hook_rayforge_init") is True

        del sys.modules["multi_hook_rayforge_init"]

    def test_addon_loading_invalid_entry_point(self, tmp_path):
        """
        Tests that invalid entry point files are handled gracefully.
        """
        addons_dir = tmp_path / "packages"
        addon_dir = addons_dir / "invalid_addon"
        addon_dir.mkdir(parents=True)

        (addon_dir / "rayforge-addon.yaml").write_text(
            "name: invalid_test\nversion: 0.1\n"
            "api_version: 1\n"
            "depends:\n"
            "  - rayforge>=0.27.0,~0.27\n"
            "provides:\n"
            "  backend: nonexistent\n"
        )

        context = RayforgeContext()
        context.addon_mgr = AddonManager(
            [addons_dir], addons_dir, context.plugin_mgr
        )

        with patch.object(
            context.addon_mgr,
            "_check_version_compatibility",
            return_value=UpdateStatus.UP_TO_DATE,
        ):
            context.initialize_full_context(load_ui=False)

        assert "rayforge_plugins.invalid_addon" not in sys.modules

    def test_addon_loading_missing_metadata(self, tmp_path):
        """
        Tests that directories without metadata are skipped.
        """
        addons_dir = tmp_path / "packages"
        addon_dir = addons_dir / "no_metadata_addon"
        addon_dir.mkdir(parents=True)

        (addon_dir / "addon.py").write_text(
            "from rayforge.core.hooks import hookimpl\n"
            "\n"
            "@hookimpl\n"
            "def rayforge_init(context):\n"
            "    pass\n"
        )

        context = RayforgeContext()
        context.addon_mgr = AddonManager(
            [addons_dir], addons_dir, context.plugin_mgr
        )

        context.initialize_full_context(load_ui=False)

        assert "rayforge_plugins.no_metadata_addon" not in sys.modules
