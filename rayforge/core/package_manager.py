import sys
import logging
import importlib.util
import urllib.request
import shutil
import tempfile
import semver
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse
import yaml
from ..config import PACKAGE_REGISTRY_URL
from .package import Package, PackageMetadata, PackageValidationError

logger = logging.getLogger(__name__)


class UpdateStatus(Enum):
    """Represents the installation status of a package from the registry."""

    NOT_INSTALLED = auto()
    UPDATE_AVAILABLE = auto()
    UP_TO_DATE = auto()


class PackageManager:
    """
    Manages the lifecycle of Rayforge packages (install, load, list).
    """

    def __init__(self, packages_dir: Path, plugin_mgr):
        """
        Args:
            packages_dir (Path): The directory where packages are installed.
            plugin_mgr: The core plugin manager instance for registration.
        """
        self.packages_dir = packages_dir
        self.plugin_mgr = plugin_mgr
        # Registry of loaded Package objects, keyed by package name
        self.loaded_packages: Dict[str, Package] = {}

    def _parse_registry_dict(
        self, registry_data: Dict[str, Any]
    ) -> List[PackageMetadata]:
        """Helper to parse the standard dictionary-based registry format."""
        packages = registry_data.get("packages", {})
        if not isinstance(packages, dict):
            logger.warning("Registry 'packages' key is not a dictionary.")
            return []

        result = []
        for pkg_id, pkg_data in packages.items():
            if not isinstance(pkg_data, dict):
                logger.warning(f"Registry entry for '{pkg_id}' is not a dict.")
                continue
            try:
                # Delegate parsing to PackageMetadata class
                meta = PackageMetadata.from_registry_entry(pkg_id, pkg_data)
                result.append(meta)
            except Exception as e:
                logger.warning(
                    f"Failed to parse registry entry '{pkg_id}': {e}"
                )
        return result

    def fetch_registry(self) -> List[PackageMetadata]:
        """
        Fetches and parses the package registry from the remote repository.
        Returns a list of PackageMetadata objects.
        """
        if yaml is None:
            logger.error("PyYAML is required to fetch the registry.")
            return []

        try:
            logger.info(f"Fetching registry from {PACKAGE_REGISTRY_URL}")
            with urllib.request.urlopen(
                PACKAGE_REGISTRY_URL, timeout=10
            ) as response:
                if response.status != 200:
                    logger.error(
                        f"Registry fetch failed: HTTP {response.status}"
                    )
                    return []
                data = response.read()
                parsed = yaml.safe_load(data)
        except Exception as e:
            logger.error(f"Failed to fetch or parse registry: {e}")
            return []

        result: List[PackageMetadata] = []
        if isinstance(parsed, list):
            # Assumes a simple list of package dicts
            for pkg_data in parsed:
                pkg_id = pkg_data.get("name")
                if not pkg_id:
                    logger.warning(
                        f"Skipping list-based registry entry without a "
                        f"'name': {pkg_data}",
                    )
                    continue
                try:
                    meta = PackageMetadata.from_registry_entry(
                        pkg_id, pkg_data
                    )
                    result.append(meta)
                except Exception as e:
                    logger.warning(
                        "Failed to parse list-based registry entry '%s': %s",
                        pkg_id,
                        e,
                    )
            return result

        if isinstance(parsed, dict):
            return self._parse_registry_dict(parsed)

        logger.warning(
            "Registry format is not a recognized list or dictionary."
        )
        return []

    def get_installed_package(self, package_id: str) -> Optional[Package]:
        """
        Finds an installed package by its canonical ID.

        Returns:
            The Package object if found, otherwise None.
        """
        return self.loaded_packages.get(package_id)

    def check_update_status(
        self, remote_meta: PackageMetadata
    ) -> Tuple[UpdateStatus, Optional[str]]:
        """
        Checks a remote package against local installations.

        Returns:
            A tuple of (UpdateStatus, local_version_str).
        """
        installed_pkg = self.get_installed_package(remote_meta.name)
        if not installed_pkg:
            return (UpdateStatus.NOT_INSTALLED, None)

        local_version = installed_pkg.metadata.version
        is_newer = self._is_newer_version(remote_meta.version, local_version)

        if is_newer:
            return (UpdateStatus.UPDATE_AVAILABLE, local_version)
        else:
            return (UpdateStatus.UP_TO_DATE, local_version)

    def _is_newer_version(self, remote_str: str, local_str: str) -> bool:
        """Compares two version strings using semver."""
        try:
            # Strip leading 'v' if present for validation
            remote_v = semver.VersionInfo.parse(remote_str.lstrip("v"))
            local_v = semver.VersionInfo.parse(local_str.lstrip("v"))
            return remote_v > local_v
        except ValueError:
            # Fallback to string comparison if not valid semver
            logger.warning(
                f"Could not parse versions '{remote_str}' or '{local_str}' "
                "with semver. Falling back to string comparison."
            )
            return remote_str != local_str

    def load_installed_packages(self):
        """Scans the packages directory and loads valid packages."""
        if not self.packages_dir.exists():
            self.packages_dir.mkdir(parents=True, exist_ok=True)
            return

        logger.info(f"Scanning for packages in {self.packages_dir}...")
        for child in self.packages_dir.iterdir():
            if child.is_dir():
                self.load_package(child)

    def load_package(self, package_path: Path):
        """
        Loads a single package from a directory.
        """
        try:
            pkg = Package.load_from_directory(package_path)

            if not pkg.metadata.provides.code:
                self.loaded_packages[pkg.metadata.name] = pkg
                logger.info(f"Loaded asset package: {pkg.metadata.name}")
                return

            self._import_and_register(pkg)

        except (PackageValidationError, FileNotFoundError) as e:
            logger.warning(
                f"Skipping invalid package at {package_path.name}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Failed to load package {package_path.name}: {e}",
                exc_info=True,
            )

    def _import_and_register(self, pkg: Package):
        """
        Imports the module specified in the package and registers it.
        """
        entry_point = pkg.metadata.provides.code
        if not entry_point:
            return

        name = pkg.metadata.name
        module_name = f"rayforge_plugins.{name}"

        # Resolve physical file path for importlib
        if ":" in entry_point:
            rel_path = entry_point.split(":")[0].replace(".", "/")
            module_path = pkg.root_path / rel_path
            module_path = (
                (module_path / "__init__.py")
                if module_path.is_dir()
                else module_path.with_suffix(".py")
            )
        else:
            module_path = pkg.root_path / entry_point

        if not module_path.exists():
            logger.error(f"Entry point {module_path} not found for {name}.")
            return

        try:
            spec = importlib.util.spec_from_file_location(
                module_name, module_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                self.plugin_mgr.register(module)
                self.loaded_packages[name] = pkg
                logger.info(f"Loaded plugin: {name} v{pkg.metadata.version}")
        except Exception as e:
            logger.error(f"Error importing plugin {name}: {e}")

    def install_package(
        self, git_url: str, package_id: Optional[str] = None
    ) -> Optional[Path]:
        """
        Install a package from a remote Git repository.

        Args:
            git_url (str): The URL of the repository to clone.
            package_id (Optional[str]): The canonical ID for the package,
                provided by the registry. If None, it's derived from the URL
                (for manual installs).
        """
        try:
            # Use importlib to avoid making `git` a hard dependency
            importlib.import_module("git")
        except ImportError:
            logger.error("GitPython is required for package installation.")
            return None

        from git import Repo

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Cloning {git_url} to staging area...")

            try:
                Repo.clone_from(git_url, temp_path, depth=1)
            except Exception as e:
                logger.error(f"Git clone failed: {e}")
                return None

            try:
                logger.info("Validating package structure and code safety...")
                pkg = Package.load_from_directory(temp_path)
                pkg.validate()
                logger.info("Validation passed.")

                # The canonical ID comes from the registry if available.
                # Fallback to repo name for manual installs.
                install_name = package_id or self._extract_repo_name(git_url)
                final_path = self.packages_dir / install_name

                if final_path.exists():
                    logger.info(f"Upgrading existing package at {final_path}")
                    # Perform a full uninstall of the old version first
                    self.uninstall_package(install_name)

                shutil.copytree(temp_path, final_path, dirs_exist_ok=True)
                git_folder = final_path / ".git"
                if git_folder.exists():
                    shutil.rmtree(git_folder, ignore_errors=True)

                logger.info(f"Successfully installed package to {final_path}")
                self.load_package(final_path)
                return final_path

            except PackageValidationError as e:
                logger.error(f"Package validation failed: {e}")
                return None
            except Exception as e:
                logger.error(f"Installation failed: {e}", exc_info=True)
                return None

    def uninstall_package(self, package_name: str) -> bool:
        """
        Deletes the package directory and unloads the module.
        """
        pkg = self.loaded_packages.get(package_name)
        if not pkg:
            logger.warning(
                f"Attempted to uninstall unknown or already "
                f"uninstalled package: {package_name}"
            )
            # If the package isn't loaded but the directory might exist,
            # we can still try to clean up the files.
            package_path = self.packages_dir / package_name
            if package_path.exists():
                self._cleanup_directory(package_path)
                return True
            return False

        package_path = pkg.root_path
        module_name = f"rayforge_plugins.{package_name}"

        try:
            # 1. Delete files from disk
            if package_path.exists() and package_path.is_dir():
                self._cleanup_directory(package_path)
                logger.info(f"Uninstalled package at {package_path}")

            # 2. Unregister from the plugin manager
            if module_name in sys.modules:
                module = sys.modules[module_name]
                self.plugin_mgr.unregister(module)

            # 3. Unload module from Python's cache
            if module_name in sys.modules:
                del sys.modules[module_name]
                logger.info(f"Unloaded module: {module_name}")

            # 4. Unregister from package manager state
            if package_name in self.loaded_packages:
                del self.loaded_packages[package_name]

            return True

        except Exception as e:
            logger.error(f"Failed to uninstall {package_name}: {e}")
            return False

    def _extract_repo_name(self, git_url: str) -> str:
        """
        Extract the repository name from a Git URL.
        """
        parsed = urlparse(git_url)
        path = parsed.path
        repo_name = path.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        return repo_name

    def _cleanup_directory(self, package_path: Path):
        """
        Clean up a directory.
        """
        try:
            if package_path.exists():
                shutil.rmtree(package_path)
                logger.debug(f"Cleaned up directory: {package_path}")
        except Exception as e:
            logger.error(f"Failed to clean up {package_path}: {e}")
