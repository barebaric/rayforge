import sys
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import Mock, MagicMock, patch
import pytest
from rayforge.package_mgr.package_manager import (
    PackageManager,
    UpdateStatus,
    AddonState,
)
from rayforge.package_mgr.package import (
    Package,
    PackageValidationError,
    PackageMetadata,
)
from rayforge.core.addon_config import (
    AddonConfig,
    AddonState as ConfigAddonState,
)
from rayforge.shared.util.versioning import parse_requirement


@pytest.fixture
def manager():
    """Provides a PackageManager instance with a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        packages_dir = Path(temp_dir) / "packages"
        plugin_mgr = Mock()
        yield PackageManager([packages_dir], packages_dir, plugin_mgr)


# Helper to create a properly structured mock package
def create_mock_package(
    name: str = "test_plugin",
    version: str = "1.0.0",
    code: Optional[str] = "plugin.py:main",
    depends: Optional[List] = None,
    requires: Optional[List] = None,
) -> MagicMock:
    """Creates a MagicMock that accurately mimics a Package object."""
    if depends is None:
        depends = ["rayforge>=0.27.0,~0.27"]
    mock_pkg = MagicMock(spec=Package)
    mock_pkg.metadata = MagicMock(spec=PackageMetadata)
    mock_pkg.metadata.name = name
    mock_pkg.metadata.version = version
    mock_pkg.metadata.depends = depends
    mock_pkg.metadata.requires = requires or []
    mock_pkg.metadata.provides = MagicMock()
    mock_pkg.metadata.provides.code = code
    mock_pkg.root_path = MagicMock(spec=Path)
    return mock_pkg


class TestPackageManagerLoading:
    """Tests related to loading existing packages."""

    def test_load_installed_packages_creates_directory(self, manager):
        manager.load_installed_packages()
        assert manager.install_dir.exists()

    def test_load_installed_packages_scans_directories(self, manager):
        manager.install_dir.mkdir()
        (manager.install_dir / "pkg1").mkdir()
        (manager.install_dir / "pkg2").mkdir()
        (manager.install_dir / "a_file.txt").touch()

        with patch.object(manager, "load_package") as mock_load:
            manager.load_installed_packages()
            assert mock_load.call_count == 2
            pkg1_path = (manager.install_dir / "pkg1").resolve()
            pkg2_path = (manager.install_dir / "pkg2").resolve()
            mock_load.assert_any_call(pkg1_path)
            mock_load.assert_any_call(pkg2_path)

    def test_load_installed_packages_scans_builtin_and_external(self):
        """Test that both builtin and external directories are scanned."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            packages_dir = temp_path / "packages"
            builtin_dir = temp_path / "builtin"

            packages_dir.mkdir()
            builtin_dir.mkdir()
            (packages_dir / "external_pkg").mkdir()
            (builtin_dir / "builtin_pkg").mkdir()

            plugin_mgr = Mock()
            manager = PackageManager(
                [builtin_dir, packages_dir], packages_dir, plugin_mgr
            )

            with patch.object(manager, "load_package") as mock_load:
                manager.load_installed_packages()
                assert mock_load.call_count == 2
                external_path = (packages_dir / "external_pkg").resolve()
                builtin_path = (builtin_dir / "builtin_pkg").resolve()
                mock_load.assert_any_call(builtin_path)
                mock_load.assert_any_call(external_path)

    def test_load_installed_packages_missing_dir(self, manager):
        """Test that missing dirs are handled gracefully."""
        nonexistent = Path("/nonexistent/packages")
        manager.package_dirs = [nonexistent, manager.install_dir]
        manager.install_dir.mkdir()
        (manager.install_dir / "pkg1").mkdir()

        with patch.object(manager, "load_package") as mock_load:
            manager.load_installed_packages()
            assert mock_load.call_count == 1

    def test_load_package_no_metadata_file(self, manager):
        package_dir = manager.install_dir / "no_meta_pkg"
        package_dir.mkdir(parents=True)
        manager.load_package(package_dir)
        assert not manager.loaded_packages
        manager.plugin_mgr.register.assert_not_called()

    @patch("rayforge.package_mgr.package_manager.Package.load_from_directory")
    def test_load_package_success(self, mock_load, manager):
        package_dir = manager.install_dir / "test_pkg"

        mock_pkg = create_mock_package(
            name="test_plugin", code="plugin.py:main"
        )
        mock_load.return_value = mock_pkg

        with (
            patch("rayforge.package_mgr.package_manager.importlib.util"),
            patch.object(
                manager,
                "_check_version_compatibility",
                return_value=UpdateStatus.UP_TO_DATE,
            ),
        ):
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
            manager.load_package(manager.install_dir / "test_pkg")

        assert "test_plugin" not in manager.loaded_packages
        assert "test_plugin" in manager.incompatible_packages


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
            if not manager.install_dir.exists():
                manager.install_dir.mkdir()
            assert not any(manager.install_dir.iterdir())

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
            if not manager.install_dir.exists():
                manager.install_dir.mkdir()
            assert not any(manager.install_dir.iterdir())

    def test_install_package_upgrades_existing(self, manager):
        manager.install_dir.mkdir()
        install_name = "my-plugin"
        git_url = f"https://example.com/repo/{install_name}.git"
        final_path = manager.install_dir / install_name
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
        manager.install_dir.mkdir()
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

            final_path = manager.install_dir / install_name
            assert result == final_path
            mock_load_package.assert_called_once_with(final_path)


class TestPackageManagerUninstall:
    """Tests for the uninstall_package method."""

    def test_uninstall_package_success(self, manager):
        """Test a successful package uninstall cleans up everything."""
        pkg_name = "my-uninstall-pkg"
        module_name = f"rayforge_plugins.{pkg_name}"

        # 1. Setup filesystem
        pkg_path = manager.install_dir / pkg_name
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

    def test_uninstall_incompatible_package(self, manager):
        """Test uninstalling an incompatible package."""
        pkg_name = "incompatible-pkg"
        pkg_path = manager.install_dir / pkg_name
        pkg_path.mkdir(parents=True)

        mock_pkg = create_mock_package(name=pkg_name)
        mock_pkg.root_path = pkg_path
        manager.incompatible_packages[pkg_name] = mock_pkg

        result = manager.uninstall_package(pkg_name)

        assert result is True
        assert not pkg_path.exists()
        assert pkg_name not in manager.incompatible_packages


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


class TestPackageManagerDisabledAddons:
    """Tests for disabled addon functionality."""

    @pytest.fixture
    def manager_with_config(self):
        """Provides a PackageManager with AddonConfig."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            packages_dir = temp_path / "packages"
            addon_config = AddonConfig(temp_path)
            addon_config.load()
            plugin_mgr = Mock()
            manager = PackageManager(
                [packages_dir], packages_dir, plugin_mgr, addon_config
            )
            yield manager

    @patch("rayforge.package_mgr.package_manager.Package.load_from_directory")
    def test_load_package_skips_disabled_addon(
        self, mock_load, manager_with_config
    ):
        """Test that a disabled addon is not loaded."""
        package_dir = manager_with_config.install_dir / "test_pkg"
        mock_pkg = create_mock_package(
            name="disabled_plugin", code="plugin.py:main"
        )
        mock_load.return_value = mock_pkg

        manager_with_config.addon_config.set_state(
            "disabled_plugin", ConfigAddonState.DISABLED
        )

        with patch.object(
            manager_with_config,
            "_check_version_compatibility",
            return_value=UpdateStatus.UP_TO_DATE,
        ):
            manager_with_config.load_package(package_dir)

        assert "disabled_plugin" not in manager_with_config.loaded_packages
        assert "disabled_plugin" in manager_with_config.disabled_packages
        manager_with_config.plugin_mgr.register.assert_not_called()

    @patch("rayforge.package_mgr.package_manager.Package.load_from_directory")
    def test_load_package_loads_enabled_addon(
        self, mock_load, manager_with_config
    ):
        """Test that an enabled addon is loaded."""
        package_dir = manager_with_config.install_dir / "test_pkg"
        mock_pkg = create_mock_package(
            name="enabled_plugin", code="plugin.py:main"
        )
        mock_load.return_value = mock_pkg

        with (
            patch("rayforge.package_mgr.package_manager.importlib.util"),
            patch.object(
                manager_with_config,
                "_check_version_compatibility",
                return_value=UpdateStatus.UP_TO_DATE,
            ),
        ):
            manager_with_config.load_package(package_dir)

        assert "enabled_plugin" in manager_with_config.loaded_packages
        manager_with_config.plugin_mgr.register.assert_called_once()

    @patch("rayforge.package_mgr.package_manager.Package.load_from_directory")
    def test_disable_addon(self, mock_load, manager_with_config):
        """Test disabling a loaded addon."""
        package_dir = manager_with_config.install_dir / "test_pkg"
        mock_pkg = create_mock_package(
            name="to_disable", code="plugin.py:main"
        )
        mock_load.return_value = mock_pkg

        with (
            patch("rayforge.package_mgr.package_manager.importlib.util"),
            patch.object(
                manager_with_config,
                "_check_version_compatibility",
                return_value=UpdateStatus.UP_TO_DATE,
            ),
        ):
            manager_with_config.load_package(package_dir)

        assert "to_disable" in manager_with_config.loaded_packages

        module_name = "rayforge_plugins.to_disable"
        sys.modules[module_name] = Mock()

        result = manager_with_config.disable_addon("to_disable")

        assert result is True
        assert "to_disable" not in manager_with_config.loaded_packages
        assert "to_disable" in manager_with_config.disabled_packages
        assert (
            manager_with_config.addon_config.get_state("to_disable")
            == ConfigAddonState.DISABLED
        )

        sys.modules.pop(module_name, None)

    @patch("rayforge.package_mgr.package_manager.Package.load_from_directory")
    def test_enable_addon(self, mock_load, manager_with_config):
        """Test enabling a disabled addon."""
        package_dir = manager_with_config.install_dir / "test_pkg"
        mock_pkg = create_mock_package(name="to_enable", code="plugin.py:main")
        mock_load.return_value = mock_pkg

        manager_with_config.addon_config.set_state(
            "to_enable", ConfigAddonState.DISABLED
        )

        with patch.object(
            manager_with_config,
            "_check_version_compatibility",
            return_value=UpdateStatus.UP_TO_DATE,
        ):
            manager_with_config.load_package(package_dir)

        assert "to_enable" in manager_with_config.disabled_packages

        orig_import = manager_with_config._import_and_register

        def fake_import_and_register(pkg):
            manager_with_config.loaded_packages[pkg.metadata.name] = pkg

        manager_with_config._import_and_register = fake_import_and_register
        try:
            with patch.object(
                manager_with_config,
                "_check_version_compatibility",
                return_value=UpdateStatus.UP_TO_DATE,
            ):
                result = manager_with_config.enable_addon("to_enable")
        finally:
            manager_with_config._import_and_register = orig_import

        assert result is True
        assert "to_enable" not in manager_with_config.disabled_packages
        assert "to_enable" in manager_with_config.loaded_packages
        assert (
            manager_with_config.addon_config.get_state("to_enable")
            == ConfigAddonState.ENABLED
        )

    def test_is_addon_enabled(self, manager_with_config):
        """Test checking if addon is enabled."""
        assert manager_with_config.is_addon_enabled("nonexistent") is False

        manager_with_config.loaded_packages["test_addon"] = Mock()
        assert manager_with_config.is_addon_enabled("test_addon") is True

    def test_get_addon_state(self, manager_with_config):
        """Test getting addon state."""
        assert (
            manager_with_config.get_addon_state("nonexistent")
            == "not_installed"
        )

        manager_with_config.loaded_packages["loaded"] = Mock()
        assert manager_with_config.get_addon_state("loaded") == "enabled"

        manager_with_config.disabled_packages["disabled"] = Mock()
        assert manager_with_config.get_addon_state("disabled") == "disabled"

        manager_with_config.incompatible_packages["incomp"] = Mock()
        assert manager_with_config.get_addon_state("incomp") == "incompatible"

    def test_uninstall_disabled_package(self, manager_with_config):
        """Test uninstalling a disabled package."""
        pkg_name = "disabled-pkg"
        pkg_path = manager_with_config.install_dir / pkg_name
        pkg_path.mkdir(parents=True)

        mock_pkg = create_mock_package(name=pkg_name)
        mock_pkg.root_path = pkg_path
        manager_with_config.disabled_packages[pkg_name] = mock_pkg

        result = manager_with_config.uninstall_package(pkg_name)

        assert result is True
        assert not pkg_path.exists()
        assert pkg_name not in manager_with_config.disabled_packages


class TestAddonStateEnum:
    """Tests for AddonState enum."""

    def test_enabled_value(self):
        assert AddonState.ENABLED.value == "enabled"

    def test_disabled_value(self):
        assert AddonState.DISABLED.value == "disabled"

    def test_pending_unload_value(self):
        assert AddonState.PENDING_UNLOAD.value == "pending_unload"

    def test_load_error_value(self):
        assert AddonState.LOAD_ERROR.value == "load_error"

    def test_not_installed_value(self):
        assert AddonState.NOT_INSTALLED.value == "not_installed"

    def test_incompatible_value(self):
        assert AddonState.INCOMPATIBLE.value == "incompatible"


class TestPackageManagerDeferredUnload:
    """Tests for deferred addon unloading when jobs are active."""

    @pytest.fixture
    def manager_with_job_callback(self):
        """Provides a PackageManager with job callback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            packages_dir = temp_path / "packages"
            addon_config = AddonConfig(temp_path)
            addon_config.load()
            plugin_mgr = Mock()

            class JobCallback:
                def __init__(self):
                    self.active = False

            job_callback = JobCallback()

            manager = PackageManager(
                [packages_dir],
                packages_dir,
                plugin_mgr,
                addon_config,
                is_job_active_callback=lambda: job_callback.active,
            )
            manager._job_callback = job_callback  # type: ignore
            yield manager

    def test_disable_addon_deferred_when_job_active(
        self, manager_with_job_callback
    ):
        """Test that addon disable is deferred when jobs are active."""
        manager_with_job_callback._job_callback.active = True
        manager_with_job_callback.loaded_packages["test_addon"] = Mock()

        result = manager_with_job_callback.disable_addon("test_addon")

        assert result is False
        assert "test_addon" in manager_with_job_callback._pending_unloads
        assert "test_addon" in manager_with_job_callback.loaded_packages
        assert (
            manager_with_job_callback.get_addon_state("test_addon")
            == "pending_unload"
        )

    def test_disable_addon_immediate_when_no_job(
        self, manager_with_job_callback
    ):
        """Test that addon disable is immediate when no jobs are active."""
        manager_with_job_callback._job_callback.active = False
        mock_pkg = Mock()
        mock_pkg.root_path = manager_with_job_callback.install_dir
        manager_with_job_callback.loaded_packages["test_addon"] = mock_pkg

        result = manager_with_job_callback.disable_addon("test_addon")

        assert result is True
        assert "test_addon" not in manager_with_job_callback._pending_unloads
        assert "test_addon" not in manager_with_job_callback.loaded_packages
        assert "test_addon" in manager_with_job_callback.disabled_packages

    def test_complete_pending_unloads(self, manager_with_job_callback):
        """Test completing pending unloads."""
        manager_with_job_callback._job_callback.active = True
        mock_pkg = Mock()
        mock_pkg.root_path = manager_with_job_callback.install_dir
        manager_with_job_callback.loaded_packages["test_addon"] = mock_pkg

        manager_with_job_callback.disable_addon("test_addon")
        assert manager_with_job_callback.has_pending_unloads()

        manager_with_job_callback._job_callback.active = False
        unloaded = manager_with_job_callback.complete_pending_unloads()

        assert unloaded == ["test_addon"]
        assert not manager_with_job_callback.has_pending_unloads()
        assert "test_addon" in manager_with_job_callback.disabled_packages

    def test_has_pending_unloads(self, manager_with_job_callback):
        """Test checking for pending unloads."""
        assert not manager_with_job_callback.has_pending_unloads()

        manager_with_job_callback._pending_unloads.add("some_addon")
        assert manager_with_job_callback.has_pending_unloads()

    def test_get_pending_unloads(self, manager_with_job_callback):
        """Test getting set of pending unloads."""
        manager_with_job_callback._pending_unloads.add("addon1")
        manager_with_job_callback._pending_unloads.add("addon2")

        pending = manager_with_job_callback.get_pending_unloads()

        assert pending == {"addon1", "addon2"}
        assert pending is not manager_with_job_callback._pending_unloads

    def test_get_addon_error(self, manager_with_job_callback):
        """Test getting load error for addon."""
        assert manager_with_job_callback.get_addon_error("unknown") is None

        manager_with_job_callback._load_errors["failed"] = "Some error"
        assert (
            manager_with_job_callback.get_addon_error("failed") == "Some error"
        )
        assert (
            manager_with_job_callback.get_addon_state("failed") == "load_error"
        )

    def test_reload_addon_success(self, manager_with_job_callback):
        """Test reloading an addon."""
        mock_pkg = create_mock_package(name="test_addon")
        mock_pkg.root_path = manager_with_job_callback.install_dir
        manager_with_job_callback.loaded_packages["test_addon"] = mock_pkg

        def fake_import(pkg):
            manager_with_job_callback.loaded_packages[pkg.metadata.name] = pkg

        manager_with_job_callback._import_and_register = fake_import

        with patch.object(
            manager_with_job_callback,
            "_check_version_compatibility",
            return_value=UpdateStatus.UP_TO_DATE,
        ):
            result = manager_with_job_callback.reload_addon("test_addon")

        assert result is True
        assert "test_addon" in manager_with_job_callback.loaded_packages

    def test_reload_addon_fails_when_job_active(
        self, manager_with_job_callback
    ):
        """Test that reload fails when jobs are active."""
        manager_with_job_callback._job_callback.active = True
        manager_with_job_callback.loaded_packages["test_addon"] = Mock()

        result = manager_with_job_callback.reload_addon("test_addon")

        assert result is False

    def test_reload_addon_not_loaded(self, manager_with_job_callback):
        """Test that reload fails when addon not loaded."""
        result = manager_with_job_callback.reload_addon("nonexistent")
        assert result is False


class TestPackageManagerDependencies:
    """Tests for addon dependency handling."""

    @pytest.fixture
    def manager_with_deps(self):
        """Provides a PackageManager for testing dependencies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            packages_dir = temp_path / "packages"
            addon_config = AddonConfig(temp_path)
            addon_config.load()
            plugin_mgr = Mock()
            manager = PackageManager(
                [packages_dir], packages_dir, plugin_mgr, addon_config
            )
            yield manager

    def test_parse_requirement_no_version(self):
        """Test parsing requirement without version constraint."""
        name, constraint = parse_requirement("laser-essentials")
        assert name == "laser-essentials"
        assert constraint is None

    def test_parse_requirement_with_version(self):
        """Test parsing requirement with version constraint."""
        name, constraint = parse_requirement("laser-essentials>=1.0.0")
        assert name == "laser-essentials"
        assert constraint == ">=1.0.0"

    def test_find_dependents(self, manager_with_deps):
        """Test finding dependents of an addon."""
        pkg_a = create_mock_package(name="addon-a", requires=["addon-b"])
        pkg_b = create_mock_package(name="addon-b")

        manager_with_deps.loaded_packages["addon-a"] = pkg_a
        manager_with_deps.loaded_packages["addon-b"] = pkg_b

        dependents = manager_with_deps._find_dependents("addon-b")
        assert dependents == ["addon-a"]

        dependents = manager_with_deps._find_dependents("addon-a")
        assert dependents == []

    def test_can_disable_no_dependents(self, manager_with_deps):
        """Test can_disable when no dependents."""
        pkg = create_mock_package(name="standalone")
        manager_with_deps.loaded_packages["standalone"] = pkg

        can_disable, reason = manager_with_deps.can_disable("standalone")
        assert can_disable is True
        assert reason == ""

    def test_can_disable_with_dependents(self, manager_with_deps):
        """Test can_disable when there are dependents."""
        pkg_a = create_mock_package(name="addon-a", requires=["addon-b"])
        pkg_b = create_mock_package(name="addon-b")

        manager_with_deps.loaded_packages["addon-a"] = pkg_a
        manager_with_deps.loaded_packages["addon-b"] = pkg_b

        can_disable, reason = manager_with_deps.can_disable("addon-b")
        assert can_disable is False
        assert "addon-a" in reason

    def test_get_missing_dependencies(self, manager_with_deps):
        """Test getting missing dependencies."""
        pkg = create_mock_package(
            name="addon-with-deps",
            requires=["missing-addon>=1.0.0"],
        )
        manager_with_deps.disabled_packages["addon-with-deps"] = pkg

        missing = manager_with_deps.get_missing_dependencies("addon-with-deps")
        assert len(missing) == 1
        assert missing[0][0] == "missing-addon"
        assert missing[0][1] == ">=1.0.0"

    def test_get_missing_dependencies_all_present(self, manager_with_deps):
        """Test get_missing_dependencies when all deps are loaded."""
        pkg_a = create_mock_package(
            name="addon-a",
            requires=["addon-b"],
        )
        pkg_b = create_mock_package(name="addon-b")

        manager_with_deps.loaded_packages["addon-a"] = pkg_a
        manager_with_deps.loaded_packages["addon-b"] = pkg_b

        missing = manager_with_deps.get_missing_dependencies("addon-a")
        assert missing == []
