"""Tests for the PackageManager class."""

import sys
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from rayforge.core.package_manager import PackageManager


class TestPackageManager:
    """Test cases for the PackageManager class."""

    def test_manager_creation(self):
        """Test creating a PackageManager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)

            assert manager.packages_dir == packages_dir
            assert manager.plugin_mgr == plugin_mgr
            assert manager.loaded_packages == {}

    def test_load_installed_packages_creates_directory(self):
        """Test that load_installed_packages creates directory if missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            manager.load_installed_packages()

            assert packages_dir.exists()
            assert manager.loaded_packages == {}

    def test_load_installed_packages_scans_directories(self):
        """Test that load_installed_packages scans package directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)

            package1_dir = packages_dir / "package1"
            package2_dir = packages_dir / "package2"
            packages_dir.mkdir()
            package1_dir.mkdir()
            package2_dir.mkdir()

            with patch.object(manager, "load_package") as mock_load:
                manager.load_installed_packages()

                assert mock_load.call_count == 2
                mock_load.assert_any_call(package1_dir)
                mock_load.assert_any_call(package2_dir)

    def test_load_package_no_metadata_file(self):
        """Test load_package returns early when metadata file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_dir = packages_dir / "test_package"
            package_dir.mkdir(parents=True)

            manager.load_package(package_dir)

            assert manager.loaded_packages == {}
            plugin_mgr.register.assert_not_called()

    def test_load_package_with_valid_metadata(self):
        """Test load_package successfully loads a valid package."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_dir = packages_dir / "test_package"
            package_dir.mkdir(parents=True)

            metadata = {
                "name": "test_plugin",
                "entry_point": "plugin.py",
            }
            meta_file = package_dir / "rayforge_package.yaml"
            with open(meta_file, "w") as f:
                yaml.dump(metadata, f)

            entry_point = package_dir / "plugin.py"
            entry_point.write_text("# test plugin")

            with patch(
                "rayforge.core.package_manager.importlib"
            ) as mock_importlib:
                mock_spec = MagicMock()
                mock_loader = MagicMock()
                mock_spec.loader = mock_loader
                mock_importlib.util.spec_from_file_location.return_value = (
                    mock_spec
                )
                mock_importlib.util.module_from_spec.return_value = MagicMock()

                manager.load_package(package_dir)

                assert "test_package" in manager.loaded_packages
                assert manager.loaded_packages["test_package"] == metadata
                plugin_mgr.register.assert_called_once()

    def test_load_package_missing_entry_point(self):
        """Test load_package handles missing entry point file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_dir = packages_dir / "test_package"
            package_dir.mkdir(parents=True)

            metadata = {
                "name": "test_plugin",
                "entry_point": "nonexistent.py",
            }
            meta_file = package_dir / "rayforge_package.yaml"
            with open(meta_file, "w") as f:
                yaml.dump(metadata, f)

            manager.load_package(package_dir)

            assert manager.loaded_packages == {}
            plugin_mgr.register.assert_not_called()

    def test_load_package_invalid_metadata(self):
        """Test load_package handles invalid metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_dir = packages_dir / "test_package"
            package_dir.mkdir(parents=True)

            metadata = {
                "name": "test_plugin",
            }
            meta_file = package_dir / "rayforge_package.yaml"
            with open(meta_file, "w") as f:
                yaml.dump(metadata, f)

            manager.load_package(package_dir)

            assert manager.loaded_packages == {}
            plugin_mgr.register.assert_not_called()

    def test_load_package_exception_handling(self):
        """Test load_package handles exceptions gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_dir = packages_dir / "test_package"
            package_dir.mkdir(parents=True)

            metadata = {
                "name": "test_plugin",
                "entry_point": "plugin.py",
            }
            meta_file = package_dir / "rayforge_package.yaml"
            with open(meta_file, "w") as f:
                yaml.dump(metadata, f)

            entry_point = package_dir / "plugin.py"
            entry_point.write_text("# test plugin")

            with patch(
                "rayforge.core.package_manager.importlib"
            ) as mock_importlib:
                mock_importlib.util.spec_from_file_location.side_effect = (
                    Exception("Import error")
                )

                manager.load_package(package_dir)

                assert manager.loaded_packages == {}
                plugin_mgr.register.assert_not_called()

    def test_read_metadata_success(self):
        """Test _read_metadata successfully parses YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_dir = packages_dir / "test_package"
            package_dir.mkdir(parents=True)

            metadata = {
                "name": "test_plugin",
                "entry_point": "plugin.py",
            }
            meta_file = package_dir / "rayforge_package.yaml"
            with open(meta_file, "w") as f:
                yaml.dump(metadata, f)

            result = manager._read_metadata(meta_file)

            assert result == metadata

    def test_read_metadata_yaml_not_available(self):
        """Test _read_metadata handles missing PyYAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_dir = packages_dir / "test_package"
            package_dir.mkdir(parents=True)

            meta_file = package_dir / "rayforge_package.yaml"
            meta_file.write_text("test: data")

            with patch("builtins.__import__", side_effect=ImportError):
                result = manager._read_metadata(meta_file)

            assert result is None

    def test_validate_metadata_valid(self):
        """Test _validate_metadata returns True for valid metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_dir = packages_dir / "test_package"

            metadata = {
                "name": "test_plugin",
                "entry_point": "plugin.py",
            }

            result = manager._validate_metadata(metadata, package_dir)

            assert result is True

    def test_validate_metadata_none(self):
        """Test _validate_metadata returns False for None metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_dir = packages_dir / "test_package"

            result = manager._validate_metadata(None, package_dir)

            assert result is False

    def test_validate_metadata_missing_entry_point(self):
        """Test _validate_metadata returns False when entry_point missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_dir = packages_dir / "test_package"

            metadata = {
                "name": "test_plugin",
            }

            result = manager._validate_metadata(metadata, package_dir)

            assert result is False

    def test_import_and_register(self):
        """Test _import_and_register imports and registers module."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_dir = packages_dir / "test_package"
            module_path = package_dir / "plugin.py"

            metadata = {
                "name": "test_plugin",
                "entry_point": "plugin.py",
            }

            mock_module = MagicMock()
            mock_spec = MagicMock()
            mock_loader = MagicMock()
            mock_spec.loader = mock_loader

            with patch(
                "rayforge.core.package_manager.importlib"
            ) as mock_importlib:
                mock_importlib.util.spec_from_file_location.return_value = (
                    mock_spec
                )
                mock_importlib.util.module_from_spec.return_value = mock_module

                manager._import_and_register(
                    metadata, package_dir, module_path
                )

                assert "test_package" in manager.loaded_packages
                assert manager.loaded_packages["test_package"] == metadata
                plugin_mgr.register.assert_called_once_with(mock_module)

    def test_import_and_register_no_spec(self):
        """Test _import_and_register handles missing spec."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_dir = packages_dir / "test_package"
            module_path = package_dir / "plugin.py"

            metadata = {
                "name": "test_plugin",
                "entry_point": "plugin.py",
            }

            with patch(
                "rayforge.core.package_manager.importlib"
            ) as mock_importlib:
                mock_importlib.util.spec_from_file_location.return_value = None

                manager._import_and_register(
                    metadata, package_dir, module_path
                )

                assert manager.loaded_packages == {}
                plugin_mgr.register.assert_not_called()

    def test_import_and_register_no_loader(self):
        """Test _import_and_register handles missing loader."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_dir = packages_dir / "test_package"
            module_path = package_dir / "plugin.py"

            metadata = {
                "name": "test_plugin",
                "entry_point": "plugin.py",
            }

            mock_spec = MagicMock()
            mock_spec.loader = None

            with patch(
                "rayforge.core.package_manager.importlib"
            ) as mock_importlib:
                mock_importlib.util.spec_from_file_location.return_value = (
                    mock_spec
                )

                manager._import_and_register(
                    metadata, package_dir, module_path
                )

                assert manager.loaded_packages == {}
                plugin_mgr.register.assert_not_called()

    def test_install_package_git_not_available(self):
        """Test install_package returns None when GitPython not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            packages_dir.mkdir()

            with patch("builtins.__import__", side_effect=ImportError):
                result = manager.install_package(
                    "https://github.com/test/repo.git"
                )

            assert result is None

    def test_install_package_already_exists(self):
        """Test install_package returns None when package exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            packages_dir.mkdir()

            existing_package = packages_dir / "repo"
            existing_package.mkdir()

            result = manager.install_package(
                "https://github.com/test/repo.git"
            )

            assert result is None

    def test_install_package_success(self):
        """Test install_package successfully clones repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            packages_dir.mkdir()

            mock_repo = MagicMock()
            mock_git_module = MagicMock()
            mock_git_module.Repo.clone_from.return_value = mock_repo

            with patch.dict(sys.modules, {"git": mock_git_module}):
                result = manager.install_package(
                    "https://github.com/test/repo.git"
                )

            expected_path = packages_dir / "repo"
            assert result == expected_path
            mock_git_module.Repo.clone_from.assert_called_once_with(
                "https://github.com/test/repo.git", expected_path
            )

    def test_install_package_clone_failure(self):
        """Test install_package handles clone failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            packages_dir.mkdir()

            mock_git_module = MagicMock()
            mock_git_module.Repo.clone_from.side_effect = Exception(
                "Clone failed"
            )

            with patch.dict(sys.modules, {"git": mock_git_module}):
                result = manager.install_package(
                    "https://github.com/test/repo.git"
                )

            assert result is None
            assert not (packages_dir / "repo").exists()

    def test_install_package_clone_failure_no_cleanup(self):
        """Test install_package does not cleanup when dir not created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            packages_dir.mkdir()

            mock_git_module = MagicMock()
            mock_git_module.Repo.clone_from.side_effect = Exception(
                "Clone failed"
            )

            with patch.dict(sys.modules, {"git": mock_git_module}):
                with patch.object(
                    manager, "_cleanup_failed_install"
                ) as mock_cleanup:
                    result = manager.install_package(
                        "https://github.com/test/repo.git"
                    )

            assert result is None
            mock_cleanup.assert_not_called()

    def test_extract_repo_name_https(self):
        """Test _extract_repo_name extracts name from HTTPS URL."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)

            url = "https://github.com/test/repo.git"
            result = manager._extract_repo_name(url)

            assert result == "repo"

    def test_extract_repo_name_ssh(self):
        """Test _extract_repo_name extracts name from SSH URL."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)

            url = "git@github.com:test/repo.git"
            result = manager._extract_repo_name(url)

            assert result == "repo"

    def test_extract_repo_name_without_git_extension(self):
        """Test _extract_repo_name handles URL without .git extension."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)

            url = "https://github.com/test/repo"
            result = manager._extract_repo_name(url)

            assert result == "repo"

    def test_extract_repo_name_with_trailing_slash(self):
        """Test _extract_repo_name handles trailing slash."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)

            url = "https://github.com/test/repo.git/"
            result = manager._extract_repo_name(url)

            assert result == "repo"

    def test_cleanup_failed_install(self):
        """Test _cleanup_failed_install removes directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_path = packages_dir / "test_package"
            package_path.mkdir(parents=True)

            manager._cleanup_failed_install(package_path)

            assert not package_path.exists()

    def test_cleanup_failed_install_nonexistent(self):
        """Test _cleanup_failed_install handles nonexistent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_path = packages_dir / "nonexistent"

            manager._cleanup_failed_install(package_path)

            assert not package_path.exists()

    def test_cleanup_failed_install_exception(self):
        """Test _cleanup_failed_install handles exceptions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_path = packages_dir / "test_package"
            package_path.mkdir(parents=True)

            with patch("builtins.__import__", side_effect=ImportError):
                manager._cleanup_failed_install(package_path)

            assert package_path.exists()

    def test_load_package_uses_default_name(self):
        """Test load_package uses directory name when name not in metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            package_dir = packages_dir / "test_package"
            package_dir.mkdir(parents=True)

            metadata = {
                "entry_point": "plugin.py",
            }
            meta_file = package_dir / "rayforge_package.yaml"
            with open(meta_file, "w") as f:
                yaml.dump(metadata, f)

            entry_point = package_dir / "plugin.py"
            entry_point.write_text("# test plugin")

            with patch(
                "rayforge.core.package_manager.importlib"
            ) as mock_importlib:
                mock_spec = MagicMock()
                mock_loader = MagicMock()
                mock_spec.loader = mock_loader
                mock_importlib.util.spec_from_file_location.return_value = (
                    mock_spec
                )
                mock_importlib.util.module_from_spec.return_value = MagicMock()

                manager.load_package(package_dir)

                assert "test_package" in manager.loaded_packages
                plugin_mgr.register.assert_called_once()

    def test_load_installed_packages_ignores_files(self):
        """Test load_installed_packages ignores non-directory entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            packages_dir = Path(temp_dir) / "packages"
            plugin_mgr = Mock()

            manager = PackageManager(packages_dir, plugin_mgr)
            packages_dir.mkdir()

            (packages_dir / "file.txt").write_text("test")

            with patch.object(manager, "load_package") as mock_load:
                manager.load_installed_packages()

                mock_load.assert_not_called()
