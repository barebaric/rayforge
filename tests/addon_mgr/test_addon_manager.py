import sys
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import Mock, MagicMock, patch
import pytest
from rayforge.addon_mgr.addon_manager import (
    AddonManager,
    UpdateStatus,
    AddonState,
)
from rayforge.addon_mgr.addon import (
    Addon,
    AddonValidationError,
    AddonMetadata,
)
from rayforge.core.addon_config import (
    AddonConfig,
    AddonState as ConfigAddonState,
)
from rayforge.shared.util.versioning import UnknownVersion
from rayforge.shared.util.versioning import parse_requirement


@pytest.fixture
def manager():
    """Provides an AddonManager instance with a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        addons_dir = Path(temp_dir) / "addons"
        plugin_mgr = Mock()
        yield AddonManager([addons_dir], addons_dir, plugin_mgr)


def create_mock_addon(
    name: str = "test_plugin",
    version: str = "1.0.0",
    backend: Optional[str] = "plugin",
    depends: Optional[List] = None,
    requires: Optional[List] = None,
) -> MagicMock:
    """Creates a MagicMock that accurately mimics a Addon object."""
    if depends is None:
        depends = ["rayforge>=0.27.0,~0.27"]
    mock_addon = MagicMock(spec=Addon)
    mock_addon.metadata = MagicMock(spec=AddonMetadata)
    mock_addon.metadata.name = name
    mock_addon.metadata.version = version
    mock_addon.metadata.depends = depends
    mock_addon.metadata.requires = requires or []
    mock_addon.metadata.provides = MagicMock()
    mock_addon.metadata.provides.backend = backend
    mock_addon.metadata.provides.frontend = None
    mock_addon.root_path = MagicMock(spec=Path)
    return mock_addon


class TestAddonManagerLoading:
    """Tests related to loading existing addons."""

    def test_load_installed_addons_creates_directory(self, manager):
        manager.load_installed_addons()
        assert manager.install_dir.exists()

    def test_load_installed_addons_scans_directories(self, manager):
        manager.install_dir.mkdir()
        (manager.install_dir / "pkg1").mkdir()
        (manager.install_dir / "pkg2").mkdir()
        (manager.install_dir / "a_file.txt").touch()

        with patch.object(manager, "load_addon") as mock_load:
            manager.load_installed_addons()
            assert mock_load.call_count == 2
            pkg1_path = (manager.install_dir / "pkg1").resolve()
            pkg2_path = (manager.install_dir / "pkg2").resolve()
            mock_load.assert_any_call(pkg1_path, backend_only=False)
            mock_load.assert_any_call(pkg2_path, backend_only=False)

    def test_load_installed_addons_scans_builtin_and_external(self):
        """Test that both builtin and external directories are scanned."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            addons_dir = temp_path / "addons"
            builtin_dir = temp_path / "builtin"

            addons_dir.mkdir()
            builtin_dir.mkdir()
            (addons_dir / "external_pkg").mkdir()
            (builtin_dir / "builtin_pkg").mkdir()

            plugin_mgr = Mock()
            manager = AddonManager(
                [builtin_dir, addons_dir], addons_dir, plugin_mgr
            )

            with patch.object(manager, "load_addon") as mock_load:
                manager.load_installed_addons()
                assert mock_load.call_count == 2
                external_path = (addons_dir / "external_pkg").resolve()
                builtin_path = (builtin_dir / "builtin_pkg").resolve()
                mock_load.assert_any_call(builtin_path, backend_only=False)
                mock_load.assert_any_call(external_path, backend_only=False)

    def test_load_installed_addons_missing_dir(self, manager):
        """Test that missing dirs are handled gracefully."""
        nonexistent = Path("/nonexistent/addons")
        manager.addon_dirs = [nonexistent, manager.install_dir]
        manager.install_dir.mkdir()
        (manager.install_dir / "pkg1").mkdir()

        with patch.object(manager, "load_addon") as mock_load:
            manager.load_installed_addons()
            assert mock_load.call_count == 1

    def test_load_addon_no_metadata_file(self, manager):
        addon_dir = manager.install_dir / "no_meta_addon"
        addon_dir.mkdir(parents=True)
        manager.load_addon(addon_dir)
        assert not manager.loaded_addons
        manager.plugin_mgr.register.assert_not_called()

    @patch("rayforge.addon_mgr.addon_manager.Addon.load_from_directory")
    def test_load_addon_success(self, mock_load, manager):
        addon_dir = manager.install_dir / "test_pkg"

        mock_addon = create_mock_addon(name="test_plugin", backend="plugin")
        mock_load.return_value = mock_addon

        with (
            patch("rayforge.addon_mgr.addon_manager.importlib.util"),
            patch.object(
                manager,
                "_check_version_compatibility",
                return_value=UpdateStatus.UP_TO_DATE,
            ),
        ):
            manager.load_addon(addon_dir)

        mock_load.assert_called_once_with(addon_dir, version=UnknownVersion)
        assert "test_plugin" in manager.loaded_addons
        manager.plugin_mgr.register.assert_called_once()

    @patch("rayforge.addon_mgr.addon_manager.Addon.load_from_directory")
    def test_load_addon_validation_error(self, mock_load, manager):
        mock_load.side_effect = AddonValidationError("Bad format")
        manager.load_addon(Path("any/path"))
        assert not manager.loaded_addons
        manager.plugin_mgr.register.assert_not_called()

    @patch("rayforge.addon_mgr.addon_manager.Addon.load_from_directory")
    def test_load_addon_incompatible_version(self, mock_load, manager):
        mock_addon = create_mock_addon(
            name="test_plugin",
            backend="plugin",
            depends=["rayforge>=99.99.99,~99.99"],
        )
        mock_load.return_value = mock_addon

        with (
            patch("rayforge.addon_mgr.addon_manager.importlib.util"),
            patch.object(
                manager,
                "_check_version_compatibility",
                return_value=UpdateStatus.INCOMPATIBLE,
            ),
        ):
            manager.load_addon(manager.install_dir / "test_pkg")

        assert "test_plugin" not in manager.loaded_addons
        assert "test_plugin" in manager.incompatible_addons


class TestAddonManagerInstallation:
    """Tests related to installing new addons."""

    def test_install_addon_git_not_available(self, manager):
        with patch("importlib.import_module", side_effect=ImportError):
            result = manager.install_addon("some_url")
            assert result is None

    def test_install_addon_clone_failure(self, manager):
        with patch(
            "git.Repo.clone_from", side_effect=Exception("network error")
        ):
            result = manager.install_addon("some_url")
            assert result is None
            if not manager.install_dir.exists():
                manager.install_dir.mkdir()
            assert not any(manager.install_dir.iterdir())

    def test_install_addon_validation_failure(self, manager):
        mock_addon = create_mock_addon()
        mock_addon.validate.side_effect = AddonValidationError("Missing asset")

        with (
            patch("git.Repo.clone_from"),
            patch(
                "rayforge.addon_mgr.addon_manager.Addon.load_from_directory",
                return_value=mock_addon,
            ),
        ):
            result = manager.install_addon("some_url")

            assert result is None
            if not manager.install_dir.exists():
                manager.install_dir.mkdir()
            assert not any(manager.install_dir.iterdir())

    def test_install_addon_upgrades_existing(self, manager):
        manager.install_dir.mkdir()
        install_name = "my-plugin"
        git_url = f"https://example.com/repo/{install_name}.git"
        final_path = manager.install_dir / install_name
        final_path.mkdir()

        mock_addon_for_validation = create_mock_addon()

        with (
            patch("git.Repo.clone_from"),
            patch(
                "rayforge.addon_mgr.addon_manager.get_git_tag_version",
                return_value="1.0.0",
            ),
            patch(
                "rayforge.addon_mgr.addon_manager.Addon.load_from_directory",
                return_value=mock_addon_for_validation,
            ),
            patch("shutil.copytree"),
            patch.object(manager, "uninstall_addon") as mock_uninstall,
            patch.object(manager, "load_addon") as mock_load_addon,
        ):
            manager.install_addon(git_url, addon_id=install_name)

            mock_uninstall.assert_called_once_with(install_name)
            mock_load_addon.assert_called_once_with(final_path)

    def test_install_addon_success(self, manager):
        manager.install_dir.mkdir()
        git_url = "https://a.b/c.git"
        install_name = "c"

        mock_addon_for_validation = create_mock_addon()

        with (
            patch("git.Repo.clone_from"),
            patch(
                "rayforge.addon_mgr.addon_manager.get_git_tag_version",
                return_value="1.0.0",
            ),
            patch(
                "rayforge.addon_mgr.addon_manager.Addon.load_from_directory",
                return_value=mock_addon_for_validation,
            ),
            patch("shutil.copytree"),
            patch.object(manager, "load_addon") as mock_load_addon,
        ):
            result = manager.install_addon(git_url)

            final_path = manager.install_dir / install_name
            assert result == final_path
            mock_load_addon.assert_called_once_with(final_path)


class TestAddonManagerUninstall:
    """Tests for the uninstall_addon method."""

    def test_uninstall_addon_success(self, manager):
        """Test a successful addon uninstall cleans up everything."""
        pkg_name = "my-uninstall-pkg"
        module_name = f"rayforge_plugins.{pkg_name}"

        # 1. Setup filesystem
        pkg_path = manager.install_dir / pkg_name
        pkg_path.mkdir(parents=True)

        # 2. Setup sys.modules to simulate a loaded module
        sys.modules[module_name] = Mock()

        # 3. Setup manager state with the loaded addon
        mock_addon = create_mock_addon(name=pkg_name)
        mock_addon.root_path = pkg_path  # Link mock object to the real path
        manager.loaded_addons[pkg_name] = mock_addon

        # Execute the uninstall
        result = manager.uninstall_addon(pkg_name)

        # Assertions
        assert result is True
        assert not pkg_path.exists()
        assert pkg_name not in manager.loaded_addons
        assert module_name not in sys.modules
        manager.plugin_mgr.unregister.assert_called_once()

    def test_uninstall_unknown_addon(self, manager):
        """Test that uninstalling a non-existent addon fails gracefully."""
        result = manager.uninstall_addon("non-existent-pkg")
        assert result is False

    def test_uninstall_incompatible_addon(self, manager):
        """Test uninstalling an incompatible addon."""
        pkg_name = "incompatible-pkg"
        pkg_path = manager.install_dir / pkg_name
        pkg_path.mkdir(parents=True)

        mock_addon = create_mock_addon(name=pkg_name)
        mock_addon.root_path = pkg_path
        manager.incompatible_addons[pkg_name] = mock_addon

        result = manager.uninstall_addon(pkg_name)

        assert result is True
        assert not pkg_path.exists()
        assert pkg_name not in manager.incompatible_addons


class TestAddonManagerUpdates:
    """Tests for the check_update_status method."""

    def test_status_not_installed(self, manager):
        remote_meta = AddonMetadata(
            "new-pkg", "", "1.0.0", ["rayforge>=0.27.0,~0.27"], Mock(), Mock()
        )
        status, local_ver = manager.check_update_status(remote_meta)
        assert status == UpdateStatus.NOT_INSTALLED
        assert local_ver is None

    def test_status_up_to_date(self, manager):
        pkg_id = "installed-pkg"
        manager.loaded_addons[pkg_id] = create_mock_addon(
            name=pkg_id, version="1.2.3"
        )
        remote_meta = AddonMetadata(
            pkg_id, "", "1.2.3", ["rayforge>=0.27.0,~0.27"], Mock(), Mock()
        )
        status, local_ver = manager.check_update_status(remote_meta)
        assert status == UpdateStatus.UP_TO_DATE
        assert local_ver == "1.2.3"

    def test_status_update_available(self, manager):
        pkg_id = "installed-pkg"
        manager.loaded_addons[pkg_id] = create_mock_addon(
            name=pkg_id, version="1.2.3"
        )
        remote_meta = AddonMetadata(
            pkg_id, "", "1.3.0", ["rayforge>=0.27.0,~0.27"], Mock(), Mock()
        )
        status, local_ver = manager.check_update_status(remote_meta)
        assert status == UpdateStatus.UPDATE_AVAILABLE
        assert local_ver == "1.2.3"

    def test_status_local_is_newer(self, manager):
        pkg_id = "installed-pkg"
        manager.loaded_addons[pkg_id] = create_mock_addon(
            name=pkg_id, version="2.0.0"
        )
        remote_meta = AddonMetadata(
            pkg_id, "", "1.5.0", ["rayforge>=0.27.0,~0.27"], Mock(), Mock()
        )
        status, local_ver = manager.check_update_status(remote_meta)
        assert status == UpdateStatus.UP_TO_DATE
        assert local_ver == "2.0.0"

    def test_check_for_updates_finds_one_update(self, manager):
        # Local addons
        manager.loaded_addons["up-to-date"] = create_mock_addon(
            name="up-to-date", version="1.0.0"
        )
        outdated_pkg = create_mock_addon(name="outdated", version="1.0.0")
        manager.loaded_addons["outdated"] = outdated_pkg

        # Remote registry
        remote_meta = [
            AddonMetadata("up-to-date", "", "1.0.0", [], Mock(), Mock()),
            AddonMetadata("outdated", "", "1.1.0", [], Mock(), Mock()),
            AddonMetadata("not-installed", "", "1.0.0", [], Mock(), Mock()),
        ]

        with patch.object(manager, "fetch_registry", return_value=remote_meta):
            updates = manager.check_for_updates()

        assert len(updates) == 1
        local_pkg, remote_pkg_meta = updates[0]
        assert local_pkg is outdated_pkg
        assert remote_pkg_meta.name == "outdated"
        assert remote_pkg_meta.version == "1.1.0"

    def test_check_for_updates_no_updates(self, manager):
        manager.loaded_addons["pkg1"] = create_mock_addon(
            name="pkg1", version="2.0.0"
        )
        remote_meta = [
            AddonMetadata("pkg1", "", "2.0.0", [], Mock(), Mock()),
            AddonMetadata("pkg2", "", "1.0.0", [], Mock(), Mock()),
        ]
        with patch.object(manager, "fetch_registry", return_value=remote_meta):
            updates = manager.check_for_updates()
        assert len(updates) == 0

    def test_check_for_updates_fetch_fails(self, manager):
        manager.loaded_addons["pkg1"] = create_mock_addon(
            name="pkg1", version="1.0.0"
        )
        with patch.object(
            manager, "fetch_registry", side_effect=Exception("Network error")
        ):
            updates = manager.check_for_updates()
        assert len(updates) == 0

    def test_check_for_updates_empty_registry(self, manager):
        manager.loaded_addons["pkg1"] = create_mock_addon(
            name="pkg1", version="1.0.0"
        )
        with patch.object(manager, "fetch_registry", return_value=[]):
            updates = manager.check_for_updates()
        assert len(updates) == 0

    def test_check_for_updates_local_not_in_registry(self, manager):
        manager.loaded_addons["local-only"] = create_mock_addon(
            name="local-only", version="1.0.0"
        )
        remote_meta = [
            AddonMetadata("other-pkg", "", "1.0.0", [], Mock(), Mock())
        ]
        with patch.object(manager, "fetch_registry", return_value=remote_meta):
            updates = manager.check_for_updates()
        assert len(updates) == 0


class TestAddonManagerHelpers:
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


class TestAddonManagerDisabledAddons:
    """Tests for disabled addon functionality."""

    @pytest.fixture
    def manager_with_config(self):
        """Provides a AddonManager with AddonConfig."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            addons_dir = temp_path / "addons"
            addon_config = AddonConfig(temp_path)
            addon_config.load()
            plugin_mgr = Mock()
            manager = AddonManager(
                [addons_dir], addons_dir, plugin_mgr, addon_config
            )
            yield manager

    @patch("rayforge.addon_mgr.addon_manager.Addon.load_from_directory")
    def test_load_addon_skips_disabled_addon(
        self, mock_load, manager_with_config
    ):
        """Test that a disabled addon is not loaded."""
        addon_dir = manager_with_config.install_dir / "test_pkg"
        mock_addon = create_mock_addon(
            name="disabled_plugin", backend="plugin"
        )
        mock_load.return_value = mock_addon

        manager_with_config.addon_config.set_state(
            "disabled_plugin", ConfigAddonState.DISABLED
        )

        with patch.object(
            manager_with_config,
            "_check_version_compatibility",
            return_value=UpdateStatus.UP_TO_DATE,
        ):
            manager_with_config.load_addon(addon_dir)

        assert "disabled_plugin" not in manager_with_config.loaded_addons
        assert "disabled_plugin" in manager_with_config.disabled_addons
        manager_with_config.plugin_mgr.register.assert_not_called()

    @patch("rayforge.addon_mgr.addon_manager.Addon.load_from_directory")
    def test_load_addon_loads_enabled_addon(
        self, mock_load, manager_with_config
    ):
        """Test that an enabled addon is loaded."""
        addon_dir = manager_with_config.install_dir / "test_pkg"
        mock_addon = create_mock_addon(name="enabled_plugin", backend="plugin")
        mock_load.return_value = mock_addon

        with (
            patch("rayforge.addon_mgr.addon_manager.importlib.util"),
            patch.object(
                manager_with_config,
                "_check_version_compatibility",
                return_value=UpdateStatus.UP_TO_DATE,
            ),
        ):
            manager_with_config.load_addon(addon_dir)

        assert "enabled_plugin" in manager_with_config.loaded_addons
        manager_with_config.plugin_mgr.register.assert_called_once()

    @patch("rayforge.addon_mgr.addon_manager.Addon.load_from_directory")
    def test_disable_addon(self, mock_load, manager_with_config):
        """Test disabling a loaded addon."""
        addon_dir = manager_with_config.install_dir / "test_pkg"
        mock_addon = create_mock_addon(name="to_disable", backend="plugin")
        mock_load.return_value = mock_addon

        with (
            patch("rayforge.addon_mgr.addon_manager.importlib.util"),
            patch.object(
                manager_with_config,
                "_check_version_compatibility",
                return_value=UpdateStatus.UP_TO_DATE,
            ),
        ):
            manager_with_config.load_addon(addon_dir)

        assert "to_disable" in manager_with_config.loaded_addons

        module_name = "rayforge_plugins.to_disable"
        sys.modules[module_name] = Mock()

        result = manager_with_config.disable_addon("to_disable")

        assert result is True
        assert "to_disable" not in manager_with_config.loaded_addons
        assert "to_disable" in manager_with_config.disabled_addons
        assert (
            manager_with_config.addon_config.get_state("to_disable")
            == ConfigAddonState.DISABLED
        )

        sys.modules.pop(module_name, None)

    @patch("rayforge.addon_mgr.addon_manager.Addon.load_from_directory")
    def test_enable_addon(self, mock_load, manager_with_config):
        """Test enabling a disabled addon."""
        addon_dir = manager_with_config.install_dir / "test_pkg"
        mock_addon = create_mock_addon(name="to_enable", backend="plugin")
        mock_load.return_value = mock_addon

        manager_with_config.addon_config.set_state(
            "to_enable", ConfigAddonState.DISABLED
        )

        with patch.object(
            manager_with_config,
            "_check_version_compatibility",
            return_value=UpdateStatus.UP_TO_DATE,
        ):
            manager_with_config.load_addon(addon_dir)

        assert "to_enable" in manager_with_config.disabled_addons

        orig_import = manager_with_config._import_and_register

        def fake_import_and_register(pkg, entry_point):
            manager_with_config.loaded_addons[pkg.metadata.name] = pkg

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
        assert "to_enable" not in manager_with_config.disabled_addons
        assert "to_enable" in manager_with_config.loaded_addons
        assert (
            manager_with_config.addon_config.get_state("to_enable")
            == ConfigAddonState.ENABLED
        )

    def test_is_addon_enabled(self, manager_with_config):
        """Test checking if addon is enabled."""
        assert manager_with_config.is_addon_enabled("nonexistent") is False

        manager_with_config.loaded_addons["test_addon"] = Mock()
        assert manager_with_config.is_addon_enabled("test_addon") is True

    def test_get_addon_state(self, manager_with_config):
        """Test getting addon state."""
        assert (
            manager_with_config.get_addon_state("nonexistent")
            == "not_installed"
        )

        manager_with_config.loaded_addons["loaded"] = Mock()
        assert manager_with_config.get_addon_state("loaded") == "enabled"

        manager_with_config.disabled_addons["disabled"] = Mock()
        assert manager_with_config.get_addon_state("disabled") == "disabled"

        manager_with_config.incompatible_addons["incomp"] = Mock()
        assert manager_with_config.get_addon_state("incomp") == "incompatible"

    def test_uninstall_disabled_addon(self, manager_with_config):
        """Test uninstalling a disabled addon."""
        pkg_name = "disabled-pkg"
        pkg_path = manager_with_config.install_dir / pkg_name
        pkg_path.mkdir(parents=True)

        mock_addon = create_mock_addon(name=pkg_name)
        mock_addon.root_path = pkg_path
        manager_with_config.disabled_addons[pkg_name] = mock_addon

        result = manager_with_config.uninstall_addon(pkg_name)

        assert result is True
        assert not pkg_path.exists()
        assert pkg_name not in manager_with_config.disabled_addons


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


class TestAddonManagerDeferredUnload:
    """Tests for deferred addon unloading when jobs are active."""

    @pytest.fixture
    def manager_with_job_callback(self):
        """Provides a AddonManager with job callback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            addons_dir = temp_path / "addons"
            addon_config = AddonConfig(temp_path)
            addon_config.load()
            plugin_mgr = Mock()

            class JobCallback:
                def __init__(self):
                    self.active = False

            job_callback = JobCallback()

            manager = AddonManager(
                [addons_dir],
                addons_dir,
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
        manager_with_job_callback.loaded_addons["test_addon"] = Mock()

        result = manager_with_job_callback.disable_addon("test_addon")

        assert result is False
        assert "test_addon" in manager_with_job_callback._pending_unloads
        assert "test_addon" in manager_with_job_callback.loaded_addons
        assert (
            manager_with_job_callback.get_addon_state("test_addon")
            == "pending_unload"
        )

    def test_disable_addon_immediate_when_no_job(
        self, manager_with_job_callback
    ):
        """Test that addon disable is immediate when no jobs are active."""
        manager_with_job_callback._job_callback.active = False
        mock_addon = Mock()
        mock_addon.root_path = manager_with_job_callback.install_dir
        manager_with_job_callback.loaded_addons["test_addon"] = mock_addon

        result = manager_with_job_callback.disable_addon("test_addon")

        assert result is True
        assert "test_addon" not in manager_with_job_callback._pending_unloads
        assert "test_addon" not in manager_with_job_callback.loaded_addons
        assert "test_addon" in manager_with_job_callback.disabled_addons

    def test_complete_pending_unloads(self, manager_with_job_callback):
        """Test completing pending unloads."""
        manager_with_job_callback._job_callback.active = True
        mock_addon = Mock()
        mock_addon.root_path = manager_with_job_callback.install_dir
        manager_with_job_callback.loaded_addons["test_addon"] = mock_addon

        manager_with_job_callback.disable_addon("test_addon")
        assert manager_with_job_callback.has_pending_unloads()

        manager_with_job_callback._job_callback.active = False
        unloaded = manager_with_job_callback.complete_pending_unloads()

        assert unloaded == ["test_addon"]
        assert not manager_with_job_callback.has_pending_unloads()
        assert "test_addon" in manager_with_job_callback.disabled_addons

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
        mock_addon = create_mock_addon(name="test_addon")
        mock_addon.root_path = manager_with_job_callback.install_dir
        manager_with_job_callback.loaded_addons["test_addon"] = mock_addon

        def fake_import(pkg, entry_point):
            manager_with_job_callback.loaded_addons[pkg.metadata.name] = pkg

        manager_with_job_callback._import_and_register = fake_import

        with patch.object(
            manager_with_job_callback,
            "_check_version_compatibility",
            return_value=UpdateStatus.UP_TO_DATE,
        ):
            result = manager_with_job_callback.reload_addon("test_addon")

        assert result is True
        assert "test_addon" in manager_with_job_callback.loaded_addons

    def test_reload_addon_fails_when_job_active(
        self, manager_with_job_callback
    ):
        """Test that reload fails when jobs are active."""
        manager_with_job_callback._job_callback.active = True
        manager_with_job_callback.loaded_addons["test_addon"] = Mock()

        result = manager_with_job_callback.reload_addon("test_addon")

        assert result is False

    def test_reload_addon_not_loaded(self, manager_with_job_callback):
        """Test that reload fails when addon not loaded."""
        result = manager_with_job_callback.reload_addon("nonexistent")
        assert result is False


class TestAddonManagerDependencies:
    """Tests for addon dependency handling."""

    @pytest.fixture
    def manager_with_deps(self):
        """Provides a AddonManager for testing dependencies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            addons_dir = temp_path / "addons"
            addon_config = AddonConfig(temp_path)
            addon_config.load()
            plugin_mgr = Mock()
            manager = AddonManager(
                [addons_dir], addons_dir, plugin_mgr, addon_config
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
        pkg_a = create_mock_addon(name="addon-a", requires=["addon-b"])
        pkg_b = create_mock_addon(name="addon-b")

        manager_with_deps.loaded_addons["addon-a"] = pkg_a
        manager_with_deps.loaded_addons["addon-b"] = pkg_b

        dependents = manager_with_deps._find_dependents("addon-b")
        assert dependents == ["addon-a"]

        dependents = manager_with_deps._find_dependents("addon-a")
        assert dependents == []

    def test_can_disable_no_dependents(self, manager_with_deps):
        """Test can_disable when no dependents."""
        pkg = create_mock_addon(name="standalone")
        manager_with_deps.loaded_addons["standalone"] = pkg

        can_disable, reason = manager_with_deps.can_disable("standalone")
        assert can_disable is True
        assert reason == ""

    def test_can_disable_with_dependents(self, manager_with_deps):
        """Test can_disable when there are dependents."""
        pkg_a = create_mock_addon(name="addon-a", requires=["addon-b"])
        pkg_b = create_mock_addon(name="addon-b")

        manager_with_deps.loaded_addons["addon-a"] = pkg_a
        manager_with_deps.loaded_addons["addon-b"] = pkg_b

        can_disable, reason = manager_with_deps.can_disable("addon-b")
        assert can_disable is False
        assert "addon-a" in reason

    def test_get_missing_dependencies(self, manager_with_deps):
        """Test getting missing dependencies."""
        pkg = create_mock_addon(
            name="addon-with-deps",
            requires=["missing-addon>=1.0.0"],
        )
        manager_with_deps.disabled_addons["addon-with-deps"] = pkg

        missing = manager_with_deps.get_missing_dependencies("addon-with-deps")
        assert len(missing) == 1
        assert missing[0][0] == "missing-addon"
        assert missing[0][1] == ">=1.0.0"

    def test_get_missing_dependencies_all_present(self, manager_with_deps):
        """Test get_missing_dependencies when all deps are loaded."""
        pkg_a = create_mock_addon(
            name="addon-a",
            requires=["addon-b"],
        )
        pkg_b = create_mock_addon(name="addon-b")

        manager_with_deps.loaded_addons["addon-a"] = pkg_a
        manager_with_deps.loaded_addons["addon-b"] = pkg_b

        missing = manager_with_deps.get_missing_dependencies("addon-a")
        assert missing == []
