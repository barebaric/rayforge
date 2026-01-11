import sys
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import Mock, MagicMock, patch
import pytest
from rayforge.package_mgr.package_manager import PackageManager, UpdateStatus
from rayforge.package_mgr.package import (
    Package,
    PackageValidationError,
    PackageMetadata,
)


@pytest.fixture
def manager():
    """Provides a PackageManager instance with a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        packages_dir = Path(temp_dir) / "packages"
        plugin_mgr = Mock()
        yield PackageManager(packages_dir, plugin_mgr)


# Helper to create a properly structured mock package
def create_mock_package(
    name: str = "test_plugin",
    version: str = "1.0.0",
    code: Optional[str] = "plugin.py:main",
    depends: Optional[List] = None,
) -> MagicMock:
    """Creates a MagicMock that accurately mimics a Package object."""
    if depends is None:
        depends = ["rayforge>=0.27.0,~0.27"]
    mock_pkg = MagicMock(spec=Package)
    # Configure the mock to have the nested metadata structure
    mock_pkg.metadata = MagicMock(spec=PackageMetadata)
    mock_pkg.metadata.name = name
    mock_pkg.metadata.version = version
    mock_pkg.metadata.depends = depends
    mock_pkg.metadata.provides = MagicMock()
    mock_pkg.metadata.provides.code = code
    mock_pkg.root_path = MagicMock(spec=Path)
    return mock_pkg


class TestPackageManagerLoading:
    """Tests related to loading existing packages."""

    def test_load_installed_packages_creates_directory(self, manager):
        manager.load_installed_packages()
        assert manager.packages_dir.exists()

    def test_load_installed_packages_scans_directories(self, manager):
        manager.packages_dir.mkdir()
        (manager.packages_dir / "pkg1").mkdir()
        (manager.packages_dir / "pkg2").mkdir()
        (manager.packages_dir / "a_file.txt").touch()

        with patch.object(manager, "load_package") as mock_load:
            manager.load_installed_packages()
            assert mock_load.call_count == 2
            pkg1_path = (manager.packages_dir / "pkg1").resolve()
            pkg2_path = (manager.packages_dir / "pkg2").resolve()
            mock_load.assert_any_call(pkg1_path)
            mock_load.assert_any_call(pkg2_path)

    def test_load_package_no_metadata_file(self, manager):
        package_dir = manager.packages_dir / "no_meta_pkg"
        package_dir.mkdir(parents=True)
        manager.load_package(package_dir)
        assert not manager.loaded_packages
        manager.plugin_mgr.register.assert_not_called()

    @patch("rayforge.package_mgr.package_manager.Package.load_from_directory")
    def test_load_package_success(self, mock_load, manager):
        package_dir = manager.packages_dir / "test_pkg"

        mock_pkg = create_mock_package(
            name="test_plugin", code="plugin.py:main"
        )
        mock_load.return_value = mock_pkg

        with patch("rayforge.package_mgr.package_manager.importlib.util"):
            manager.load_package(package_dir)

        mock_load.assert_called_once_with(package_dir)
        assert "test_plugin" in manager.loaded_packages
        manager.plugin_mgr.register.assert_called_once()

    @patch("rayforge.package_mgr.package_manager.Package.load_from_directory")
    def test_load_package_validation_error(self, mock_load, manager):
        mock_load.side_effect = PackageValidationError("Bad format")
        manager.load_package(Path("any/path"))
        assert not manager.loaded_packages
        manager.plugin_mgr.register.assert_not_called()

    @patch("rayforge.package_mgr.package_manager.Package.load_from_directory")
    def test_load_package_incompatible_version(self, mock_load, manager):
        mock_pkg = create_mock_package(
            name="test_plugin",
            code="plugin.py:main",
            depends=["rayforge>=99.99.99,~99.99"],
        )
        mock_load.return_value = mock_pkg

        with (
            patch("rayforge.package_mgr.package_manager.importlib.util"),
            patch.object(
                manager,
                "_check_version_compatibility",
                return_value=UpdateStatus.INCOMPATIBLE,
            ),
        ):
            manager.load_package(manager.packages_dir / "test_pkg")

        assert "test_plugin" not in manager.loaded_packages


class TestPackageManagerInstallation:
    """Tests related to installing new packages."""

    def test_install_package_git_not_available(self, manager):
        with patch("importlib.import_module", side_effect=ImportError):
            result = manager.install_package("some_url")
            assert result is None

    def test_install_package_clone_failure(self, manager):
        with patch(
            "git.Repo.clone_from", side_effect=Exception("network error")
        ):
            result = manager.install_package("some_url")
            assert result is None
            if not manager.packages_dir.exists():
                manager.packages_dir.mkdir()
            assert not any(manager.packages_dir.iterdir())

    def test_install_package_validation_failure(self, manager):
        mock_pkg = create_mock_package()
        mock_pkg.validate.side_effect = PackageValidationError("Missing asset")

        with (
            patch("git.Repo.clone_from"),
            patch(
                "rayforge.package_mgr.package_manager."
                "Package.load_from_directory",
                return_value=mock_pkg,
            ),
        ):
            result = manager.install_package("some_url")

            assert result is None
            if not manager.packages_dir.exists():
                manager.packages_dir.mkdir()
            assert not any(manager.packages_dir.iterdir())

    def test_install_package_upgrades_existing(self, manager):
        manager.packages_dir.mkdir()
        install_name = "my-plugin"
        git_url = f"https://example.com/repo/{install_name}.git"
        final_path = manager.packages_dir / install_name
        final_path.mkdir()  # Make it exist so the upgrade logic triggers

        # Mock for the validation step inside install_package
        mock_pkg_for_validation = create_mock_package()

        with (
            patch("git.Repo.clone_from"),
            patch(
                "rayforge.package_mgr.package_manager."
                "Package.load_from_directory",
                return_value=mock_pkg_for_validation,
            ),
            patch("shutil.copytree"),
            patch.object(manager, "uninstall_package") as mock_uninstall,
            patch.object(manager, "load_package") as mock_load_package,
        ):
            manager.install_package(git_url, package_id=install_name)

            mock_uninstall.assert_called_once_with(install_name)
            mock_load_package.assert_called_once_with(final_path)

    def test_install_package_success(self, manager):
        manager.packages_dir.mkdir()
        git_url = "https://a.b/c.git"
        install_name = "c"  # Derived from URL for manual install

        mock_pkg_for_validation = create_mock_package()

        with (
            patch("git.Repo.clone_from"),
            patch(
                "rayforge.package_mgr.package_manager."
                "Package.load_from_directory",
                return_value=mock_pkg_for_validation,
            ),
            patch("shutil.copytree"),
            patch.object(manager, "load_package") as mock_load_package,
        ):
            result = manager.install_package(git_url)

            final_path = manager.packages_dir / install_name
            assert result == final_path
            mock_load_package.assert_called_once_with(final_path)


class TestPackageManagerUninstall:
    """Tests for the uninstall_package method."""

    def test_uninstall_package_success(self, manager):
        """Test a successful package uninstall cleans up everything."""
        pkg_name = "my-uninstall-pkg"
        module_name = f"rayforge_plugins.{pkg_name}"

        # 1. Setup filesystem
        pkg_path = manager.packages_dir / pkg_name
        pkg_path.mkdir(parents=True)

        # 2. Setup sys.modules to simulate a loaded module
        sys.modules[module_name] = Mock()

        # 3. Setup manager state with the loaded package
        mock_pkg = create_mock_package(name=pkg_name)
        mock_pkg.root_path = pkg_path  # Link mock object to the real path
        manager.loaded_packages[pkg_name] = mock_pkg

        # Execute the uninstall
        result = manager.uninstall_package(pkg_name)

        # Assertions
        assert result is True
        assert not pkg_path.exists()
        assert pkg_name not in manager.loaded_packages
        assert module_name not in sys.modules
        manager.plugin_mgr.unregister.assert_called_once()

    def test_uninstall_unknown_package(self, manager):
        """Test that uninstalling a non-existent package fails gracefully."""
        result = manager.uninstall_package("non-existent-pkg")
        assert result is False


class TestPackageManagerUpdates:
    """Tests for the check_update_status method."""

    def test_status_not_installed(self, manager):
        remote_meta = PackageMetadata(
            "new-pkg", "", "1.0.0", ["rayforge>=0.27.0,~0.27"], Mock(), Mock()
        )
        status, local_ver = manager.check_update_status(remote_meta)
        assert status == UpdateStatus.NOT_INSTALLED
        assert local_ver is None

    def test_status_up_to_date(self, manager):
        pkg_id = "installed-pkg"
        manager.loaded_packages[pkg_id] = create_mock_package(
            name=pkg_id, version="1.2.3"
        )
        remote_meta = PackageMetadata(
            pkg_id, "", "1.2.3", ["rayforge>=0.27.0,~0.27"], Mock(), Mock()
        )
        status, local_ver = manager.check_update_status(remote_meta)
        assert status == UpdateStatus.UP_TO_DATE
        assert local_ver == "1.2.3"

    def test_status_update_available(self, manager):
        pkg_id = "installed-pkg"
        manager.loaded_packages[pkg_id] = create_mock_package(
            name=pkg_id, version="1.2.3"
        )
        remote_meta = PackageMetadata(
            pkg_id, "", "1.3.0", ["rayforge>=0.27.0,~0.27"], Mock(), Mock()
        )
        status, local_ver = manager.check_update_status(remote_meta)
        assert status == UpdateStatus.UPDATE_AVAILABLE
        assert local_ver == "1.2.3"

    def test_status_local_is_newer(self, manager):
        pkg_id = "installed-pkg"
        manager.loaded_packages[pkg_id] = create_mock_package(
            name=pkg_id, version="2.0.0"
        )
        remote_meta = PackageMetadata(
            pkg_id, "", "1.5.0", ["rayforge>=0.27.0,~0.27"], Mock(), Mock()
        )
        status, local_ver = manager.check_update_status(remote_meta)
        assert status == UpdateStatus.UP_TO_DATE
        assert local_ver == "2.0.0"

    def test_check_for_updates_finds_one_update(self, manager):
        # Local packages
        manager.loaded_packages["up-to-date"] = create_mock_package(
            name="up-to-date", version="1.0.0"
        )
        outdated_pkg = create_mock_package(name="outdated", version="1.0.0")
        manager.loaded_packages["outdated"] = outdated_pkg

        # Remote registry
        remote_meta = [
            PackageMetadata("up-to-date", "", "1.0.0", [], Mock(), Mock()),
            PackageMetadata("outdated", "", "1.1.0", [], Mock(), Mock()),
            PackageMetadata("not-installed", "", "1.0.0", [], Mock(), Mock()),
        ]

        with patch.object(manager, "fetch_registry", return_value=remote_meta):
            updates = manager.check_for_updates()

        assert len(updates) == 1
        local_pkg, remote_pkg_meta = updates[0]
        assert local_pkg is outdated_pkg
        assert remote_pkg_meta.name == "outdated"
        assert remote_pkg_meta.version == "1.1.0"

    def test_check_for_updates_no_updates(self, manager):
        manager.loaded_packages["pkg1"] = create_mock_package(
            name="pkg1", version="2.0.0"
        )
        remote_meta = [
            PackageMetadata("pkg1", "", "2.0.0", [], Mock(), Mock()),
            PackageMetadata("pkg2", "", "1.0.0", [], Mock(), Mock()),
        ]
        with patch.object(manager, "fetch_registry", return_value=remote_meta):
            updates = manager.check_for_updates()
        assert len(updates) == 0

    def test_check_for_updates_fetch_fails(self, manager):
        manager.loaded_packages["pkg1"] = create_mock_package(
            name="pkg1", version="1.0.0"
        )
        with patch.object(
            manager, "fetch_registry", side_effect=Exception("Network error")
        ):
            updates = manager.check_for_updates()
        assert len(updates) == 0

    def test_check_for_updates_empty_registry(self, manager):
        manager.loaded_packages["pkg1"] = create_mock_package(
            name="pkg1", version="1.0.0"
        )
        with patch.object(manager, "fetch_registry", return_value=[]):
            updates = manager.check_for_updates()
        assert len(updates) == 0

    def test_check_for_updates_local_not_in_registry(self, manager):
        manager.loaded_packages["local-only"] = create_mock_package(
            name="local-only", version="1.0.0"
        )
        remote_meta = [
            PackageMetadata("other-pkg", "", "1.0.0", [], Mock(), Mock())
        ]
        with patch.object(manager, "fetch_registry", return_value=remote_meta):
            updates = manager.check_for_updates()
        assert len(updates) == 0


class TestPackageManagerHelpers:
    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://github.com/user/repo.git", "repo"),
            ("https://github.com/user/repo", "repo"),
            ("git@github.com:user/repo.git", "repo"),
            ("https://github.com/user/repo.git/", "repo"),
        ],
    )
    def test_extract_repo_name(self, manager, url, expected):
        assert manager._extract_repo_name(url) == expected
