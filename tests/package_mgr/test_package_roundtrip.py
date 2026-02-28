"""Integration tests for package loading and execution."""

import sys
from unittest.mock import patch

from rayforge.context import RayforgeContext
from rayforge.package_mgr.package_manager import (
    PackageManager,
    UpdateStatus,
)


class TestPackageRoundTrip:
    """Test cases for full package loading lifecycle."""

    def test_full_package_loading_and_execution(self, tmp_path):
        """
        Creates a real package file structure, loads it, and ensures it runs.
        """
        packages_dir = tmp_path / "packages"
        package_dir = packages_dir / "integration_test_package"
        package_dir.mkdir(parents=True)

        (package_dir / "rayforge-package.yaml").write_text(
            "name: this_name_is_ignored\nversion: 0.1\n"
            "api_version: 1\n"
            "depends:\n"
            "  - rayforge>=0.27.0,~0.27\n"
            "entry_point: package.py\n"
        )

        (package_dir / "package.py").write_text(
            "import sys\n"
            "from rayforge.core.hooks import hookimpl\n"
            "\n"
            "sys.modules['integration_test_package_loaded'] = False\n"
            "\n"
            "@hookimpl\n"
            "def rayforge_init(context):\n"
            "    sys.modules['integration_test_package_loaded'] = True\n"
        )

        context = RayforgeContext()
        context.package_mgr = PackageManager(
            [packages_dir], packages_dir, context.plugin_mgr
        )

        with patch.object(
            context.package_mgr,
            "_check_version_compatibility",
            return_value=UpdateStatus.UP_TO_DATE,
        ):
            context.initialize_full_context()

        # The module name is derived from the directory, not the YAML file.
        assert "rayforge_plugins.integration_test_package" in sys.modules
        assert sys.modules.get("integration_test_package_loaded") is True

        del sys.modules["integration_test_package_loaded"]

    def test_package_loading_with_multiple_hooks(self, tmp_path):
        """
        Tests that a package can implement multiple hooks.
        """
        packages_dir = tmp_path / "packages"
        package_dir = packages_dir / "multi_hook_package"
        package_dir.mkdir(parents=True)

        (package_dir / "rayforge-package.yaml").write_text(
            "name: this_name_is_ignored\nversion: 0.1\n"
            "api_version: 1\n"
            "depends:\n"
            "  - rayforge>=0.27.0,~0.27\n"
            "entry_point: package.py\n"
        )

        (package_dir / "package.py").write_text(
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
        context.package_mgr = PackageManager(
            [packages_dir], packages_dir, context.plugin_mgr
        )

        with patch.object(
            context.package_mgr,
            "_check_version_compatibility",
            return_value=UpdateStatus.UP_TO_DATE,
        ):
            context.initialize_full_context()

        # The module name is derived from the directory, not the YAML file.
        assert "rayforge_plugins.multi_hook_package" in sys.modules
        assert sys.modules.get("multi_hook_rayforge_init") is True

        del sys.modules["multi_hook_rayforge_init"]

    def test_package_loading_invalid_entry_point(self, tmp_path):
        """
        Tests that invalid entry point files are handled gracefully.
        """
        packages_dir = tmp_path / "packages"
        package_dir = packages_dir / "invalid_package"
        package_dir.mkdir(parents=True)

        (package_dir / "rayforge-package.yaml").write_text(
            "name: invalid_test\nversion: 0.1\n"
            "api_version: 1\n"
            "depends:\n"
            "  - rayforge>=0.27.0,~0.27\n"
            "entry_point: nonexistent.py\n"
        )

        context = RayforgeContext()
        context.package_mgr = PackageManager(
            [packages_dir], packages_dir, context.plugin_mgr
        )

        with patch.object(
            context.package_mgr,
            "_check_version_compatibility",
            return_value=UpdateStatus.UP_TO_DATE,
        ):
            context.initialize_full_context()

        assert "rayforge_plugins.invalid_package" not in sys.modules

    def test_package_loading_missing_metadata(self, tmp_path):
        """
        Tests that directories without metadata are skipped.
        """
        packages_dir = tmp_path / "packages"
        package_dir = packages_dir / "no_metadata_package"
        package_dir.mkdir(parents=True)

        (package_dir / "package.py").write_text(
            "from rayforge.core.hooks import hookimpl\n"
            "\n"
            "@hookimpl\n"
            "def rayforge_init(context):\n"
            "    pass\n"
        )

        context = RayforgeContext()
        context.package_mgr = PackageManager(
            [packages_dir], packages_dir, context.plugin_mgr
        )

        context.initialize_full_context()

        assert "rayforge_plugins.no_metadata_package" not in sys.modules
