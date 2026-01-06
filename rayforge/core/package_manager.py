import sys
import logging
import importlib.util
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class PackageManager:
    def __init__(self, packages_dir: Path, plugin_mgr):
        self.packages_dir = packages_dir
        self.plugin_mgr = plugin_mgr
        self.loaded_packages = {}

    def load_installed_packages(self):
        """Scans the packages directory and loads valid plugins."""
        if not self.packages_dir.exists():
            self.packages_dir.mkdir(parents=True, exist_ok=True)
            return

        for child in self.packages_dir.iterdir():
            if child.is_dir():
                self.load_package(child)

    def load_package(self, package_path: Path):
        """
        Loads a single package from a directory.
        Expects a 'rayforge_package.yaml' file.
        """
        meta_file = package_path / "rayforge_package.yaml"
        if not meta_file.exists():
            return

        try:
            metadata = self._read_metadata(meta_file)
            if not self._validate_metadata(metadata, package_path):
                return

            assert metadata is not None
            entry_point_file = metadata["entry_point"]
            module_path = package_path / entry_point_file

            if not module_path.exists():
                logger.warning(f"Entry point {module_path} not found.")
                return

            self._import_and_register(metadata, package_path, module_path)

        except Exception as e:
            logger.error(
                f"Failed to load package {package_path.name}: {e}",
                exc_info=True,
            )

    def _read_metadata(self, meta_file: Path) -> Optional[dict]:
        """Reads and parses the package metadata YAML file."""
        try:
            import yaml

            with open(meta_file, "r") as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.error("PyYAML is required for plugin loading")
            return None

    def _validate_metadata(
        self, metadata: Optional[dict], package_path: Path
    ) -> bool:
        """Validates the package metadata."""
        if not metadata or "entry_point" not in metadata:
            logger.warning(
                f"Skipping invalid package at {package_path}: "
                "missing entry_point"
            )
            return False
        return True

    def _import_and_register(
        self, metadata: dict, package_path: Path, module_path: Path
    ):
        """Imports the module and registers it with the plugin manager."""
        name = metadata.get("name", package_path.name)
        module_name = f"rayforge_plugins.{name}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            self.plugin_mgr.register(module)
            self.loaded_packages[package_path.name] = metadata
            logger.info(f"Loaded plugin: {metadata.get('name')}")

    def install_package(self, git_url: str) -> Optional[Path]:
        """
        Install a package from a remote Git repository.

        Clones the repository into the packages directory.

        Args:
            git_url: URL of the Git repository to clone.

        Returns:
            Path to the cloned package directory, or None if failed.
        """
        try:
            import git
        except ImportError:
            logger.error("GitPython is required for package installation")
            return None

        repo_name = self._extract_repo_name(git_url)
        package_path = self.packages_dir / repo_name

        if package_path.exists():
            logger.warning(
                f"Package directory {package_path} already exists. "
                "Remove it first or use a different name."
            )
            return None

        try:
            logger.info(f"Cloning {git_url} to {package_path}")
            git.Repo.clone_from(git_url, package_path)
            logger.info(f"Successfully cloned package to {package_path}")
            return package_path
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}", exc_info=True)
            if package_path.exists():
                self._cleanup_failed_install(package_path)
            return None

    def _extract_repo_name(self, git_url: str) -> str:
        """
        Extract the repository name from a Git URL.

        Args:
            git_url: URL of the Git repository.

        Returns:
            Repository name without .git extension.
        """
        parsed = urlparse(git_url)
        path = parsed.path
        repo_name = path.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        return repo_name

    def _cleanup_failed_install(self, package_path: Path):
        """
        Clean up a failed installation directory.

        Args:
            package_path: Path to the failed installation directory.
        """
        try:
            import shutil

            if package_path.exists():
                shutil.rmtree(package_path)
                logger.info(f"Cleaned up failed install: {package_path}")
        except Exception as e:
            logger.error(f"Failed to clean up {package_path}: {e}")
