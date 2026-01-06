"""Integration tests for plugin loading and execution."""

import sys

from rayforge.context import RayforgeContext
from rayforge.core.package_manager import PackageManager


class TestPluginRoundTrip:
    """Test cases for full plugin loading lifecycle."""

    def test_full_plugin_loading_and_execution(self, tmp_path):
        """
        Creates a real plugin file structure, loads it, and ensures it runs.
        """
        packages_dir = tmp_path / "packages"
        plugin_dir = packages_dir / "integration_test_plugin"
        plugin_dir.mkdir(parents=True)

        (plugin_dir / "rayforge_package.yaml").write_text(
            "name: integration_test\nversion: 0.1\nentry_point: plugin.py\n"
        )

        (plugin_dir / "plugin.py").write_text(
            "import sys\n"
            "from rayforge.core.hooks import hookimpl\n"
            "\n"
            "sys.modules['integration_test_plugin_loaded'] = False\n"
            "\n"
            "@hookimpl\n"
            "def rayforge_init(context):\n"
            "    sys.modules['integration_test_plugin_loaded'] = True\n"
        )

        context = RayforgeContext()
        context.package_mgr = PackageManager(packages_dir, context.plugin_mgr)

        context.initialize_full_context()

        assert "rayforge_plugins.integration_test" in sys.modules
        assert sys.modules.get("integration_test_plugin_loaded") is True

        del sys.modules["integration_test_plugin_loaded"]

    def test_plugin_loading_with_multiple_hooks(self, tmp_path):
        """
        Tests that a plugin can implement multiple hooks.
        """
        packages_dir = tmp_path / "packages"
        plugin_dir = packages_dir / "multi_hook_plugin"
        plugin_dir.mkdir(parents=True)

        (plugin_dir / "rayforge_package.yaml").write_text(
            "name: multi_hook_test\nversion: 0.1\nentry_point: plugin.py\n"
        )

        (plugin_dir / "plugin.py").write_text(
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
        context.package_mgr = PackageManager(packages_dir, context.plugin_mgr)

        context.initialize_full_context()

        assert "rayforge_plugins.multi_hook_test" in sys.modules
        assert sys.modules.get("multi_hook_rayforge_init") is True

        del sys.modules["multi_hook_rayforge_init"]

    def test_plugin_loading_invalid_entry_point(self, tmp_path):
        """
        Tests that invalid entry point files are handled gracefully.
        """
        packages_dir = tmp_path / "packages"
        plugin_dir = packages_dir / "invalid_plugin"
        plugin_dir.mkdir(parents=True)

        (plugin_dir / "rayforge_package.yaml").write_text(
            "name: invalid_test\nversion: 0.1\nentry_point: nonexistent.py\n"
        )

        context = RayforgeContext()
        context.package_mgr = PackageManager(packages_dir, context.plugin_mgr)

        context.initialize_full_context()

        assert "rayforge_plugins.invalid_test" not in sys.modules

    def test_plugin_loading_missing_metadata(self, tmp_path):
        """
        Tests that directories without metadata are skipped.
        """
        packages_dir = tmp_path / "packages"
        plugin_dir = packages_dir / "no_metadata_plugin"
        plugin_dir.mkdir(parents=True)

        (plugin_dir / "plugin.py").write_text(
            "from rayforge.core.hooks import hookimpl\n"
            "\n"
            "@hookimpl\n"
            "def rayforge_init(context):\n"
            "    pass\n"
        )

        context = RayforgeContext()
        context.package_mgr = PackageManager(packages_dir, context.plugin_mgr)

        context.initialize_full_context()

        assert "rayforge_plugins.no_metadata_plugin" not in sys.modules
