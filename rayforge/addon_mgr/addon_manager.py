import sys
import logging
import importlib.util
import urllib.request
import shutil
import tempfile
from enum import Enum, auto
from pathlib import Path
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Tuple,
    Callable,
    Set,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
)
from urllib.parse import urlparse

import pluggy
import yaml
from blinker import Signal

from .. import __version__
from ..config import ADDON_REGISTRY_URL
from ..core.addon_config import AddonConfig, AddonState as ConfigAddonState
from ..license import LicenseValidator
from ..shared.util.po_compiler import compile_po_to_mo, needs_compilation
from ..shared.util.versioning import (
    check_rayforge_compatibility,
    get_git_tag_version,
    is_newer_version,
    parse_requirement,
    UnknownVersion,
)
from .addon import Addon, AddonMetadata, AddonValidationError
from .manifest import AddonManifest

if TYPE_CHECKING:
    from ..shared.tasker.manager import TaskManagerProxy

logger = logging.getLogger(__name__)


@runtime_checkable
class AddonRegistry(Protocol):
    """Protocol for registries that support addon item cleanup."""

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all items registered by the named addon.

        Args:
            addon_name: The canonical name of the addon.

        Returns:
            The number of items unregistered.
        """
        ...


class AddonState(Enum):
    """Represents the state of an addon."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    PENDING_UNLOAD = "pending_unload"
    LOAD_ERROR = "load_error"
    NOT_INSTALLED = "not_installed"
    INCOMPATIBLE = "incompatible"
    LICENSE_REQUIRED = "license_required"


class UpdateStatus(Enum):
    """Represents the installation status of an addon from the registry."""

    NOT_INSTALLED = auto()
    UPDATE_AVAILABLE = auto()
    UP_TO_DATE = auto()
    INCOMPATIBLE = auto()


class AddonManager:
    """
    Manages the lifecycle of Rayforge addons (install, load, list).
    """

    def __init__(
        self,
        addon_dirs: List[Path],
        install_dir: Path,
        plugin_mgr: pluggy.PluginManager,
        addon_config: Optional[AddonConfig] = None,
        is_job_active_callback: Optional[Callable[[], bool]] = None,
        registries: Optional[Dict[str, "AddonRegistry"]] = None,
        shared_state: Optional[Dict[str, Any]] = None,
        task_mgr: Optional["TaskManagerProxy"] = None,
        license_validator: Optional[LicenseValidator] = None,
    ):
        """
        Args:
            addon_dirs (List[Path]): Directories to scan for addons.
            install_dir (Path): Directory for installing new addons.
            plugin_mgr (pluggy.PluginManager): The core plugin manager
                instance for registration.
            addon_config (Optional[AddonConfig]): Addon state persistence
                manager. If None, addons will always be loaded.
            is_job_active_callback (Optional[Callable]): A callback that
                returns True if any job is currently active. Used to defer
                addon unloading until jobs complete.
            registries (Optional[Dict[str, AddonRegistry]]): Dict mapping
                hook parameter names to registry instances. Expected keys:
                'step_registry', 'producer_registry', 'widget_registry',
                'menu_registry', 'layout_registry'.
            shared_state (Optional[Any]): Shared dict for worker state,
                used to populate addon module paths for worker processes.
            task_mgr (Optional[TaskManagerProxy]): Proxy to trigger worker
                pool restarts upon configuration changes.
            license_validator (Optional[LicenseValidator]): License validator
                for checking paid addon licenses.
        """
        self.addon_dirs = addon_dirs
        self.install_dir = install_dir
        self.plugin_mgr = plugin_mgr
        self.addon_config = addon_config
        self.is_job_active_callback = is_job_active_callback
        self.registries: Dict[str, AddonRegistry] = registries or {}
        self._window: Optional[Any] = None
        self.loaded_addons: Dict[str, Addon] = {}
        self.incompatible_addons: Dict[str, Addon] = {}
        self.disabled_addons: Dict[str, Addon] = {}
        self.license_required_addons: Dict[str, Addon] = {}
        self._pending_unloads: Set[str] = set()
        self._load_errors: Dict[str, str] = {}
        self._shared_state = shared_state
        self._task_mgr = task_mgr
        self.license_validator = license_validator

        self.addon_reloaded = Signal()

        if license_validator:
            license_validator.changed.connect(self._on_license_changed)

    def _on_license_changed(self, sender):
        for addon_name in list(self.license_required_addons.keys()):
            self.recheck_license(addon_name)

    def set_registries(self, registries: Dict[str, AddonRegistry]):
        """
        Set the registries dict for addon cleanup.

        Args:
            registries: Dict mapping hook parameter names to registry
                instances. Expected keys: 'step_registry',
                'producer_registry', 'widget_registry', 'action_registry',
                'layout_registry'.
        """
        self.registries = registries

    def set_window(self, window: Any):
        """
        Set the main window for registering actions.

        Args:
            window: The MainWindow instance for registering actions.
        """
        self._window = window

    def set_task_manager(self, task_mgr: "TaskManagerProxy"):
        """
        Set the task manager proxy to trigger worker pool restarts.
        """
        self._task_mgr = task_mgr

    def set_shared_state(self, shared_state: Dict[str, Any]):
        """
        Set the shared state dict and immediately build the addon manifest.

        Args:
            shared_state: Shared dict for worker state.
        """
        self._shared_state = shared_state
        self._build_and_update_manifest()

    def _restart_workers(self):
        """
        Restarts the worker pool to ensure runtime modifications to addons
        are securely reflected in background processes.
        """
        if self._task_mgr is not None:
            logger.info("Restarting worker pool due to addon changes...")
            self._task_mgr.restart_worker_pool()

    def _build_and_update_manifest(self):
        """
        Scans all known addons to build a comprehensive manifest and
        updates the shared state for worker processes.

        The namespace structure for addons is:
        rayforge_addons.<ADDON_NAME>.<MODULE_STRUCTURE>

        Where <ADDON_NAME> is the name from the rayforge-addon.yaml.
        This guarantees isolation between addons even if they have internal
        modules with the same name.
        """
        manifest = AddonManifest()
        all_addons = (
            list(self.loaded_addons.values())
            + list(self.disabled_addons.values())
            + list(self.incompatible_addons.values())
            + list(self.license_required_addons.values())
        )

        root_ns = "rayforge_addons"
        manifest.namespaces.add(root_ns)

        for addon in all_addons:
            addon_id = addon.metadata.name

            # Root namespace for this addon
            addon_ns = f"rayforge_addons.{addon_id}"
            manifest.namespaces.add(addon_ns)

            # Scan all python files to register every potential module
            for py_file in addon.root_path.rglob("*.py"):
                # Ignore hidden files/directories
                if any(p.startswith(".") for p in py_file.parts):
                    continue

                rel_path = py_file.relative_to(addon.root_path)
                parts = rel_path.with_suffix("").parts

                if py_file.name == "__init__.py":
                    mod_parts = parts[:-1]
                else:
                    mod_parts = parts

                if not mod_parts:
                    continue

                # Ensure it's a valid python identifier sequence
                if not all(p.isidentifier() for p in mod_parts):
                    continue

                module_name = f"{addon_ns}.{'.'.join(mod_parts)}"
                manifest.module_paths[module_name] = str(py_file)

                # Add namespaces for parent packages
                for i in range(1, len(mod_parts)):
                    manifest.namespaces.add(
                        f"{addon_ns}.{'.'.join(mod_parts[:i])}"
                    )

        # Populate enabled backend entry points using the namespaced structure
        for addon in self.loaded_addons.values():
            if addon.metadata.provides.backend:
                backend_module = (
                    f"rayforge_addons.{addon.metadata.name}."
                    f"{addon.metadata.provides.backend}"
                )
                manifest.enabled_backend_modules.append(backend_module)

        # Prefer the shared state from the active task manager, as it may have
        # changed after a pool restart. Fall back to the stored shared state.
        target_state = None
        if self._task_mgr:
            target_state = self._task_mgr.get_shared_state()

        if target_state is None:
            target_state = self._shared_state

        if target_state is not None:
            logger.debug(
                f"Updating addon manifest in shared state. "
                f"{len(manifest.module_paths)} modules, "
                f"{len(manifest.enabled_backend_modules)} enabled backends."
            )
            target_state["addon_manifest"] = manifest
        else:
            logger.warning(
                "No shared state available to update addon manifest."
            )

    def _parse_registry_dict(
        self, registry_data: Dict[str, Any]
    ) -> List[AddonMetadata]:
        """Helper to parse the standard dictionary-based registry format."""
        addons = registry_data.get("addons", {})
        if not isinstance(addons, dict):
            logger.warning("Registry 'addons' key is not a dictionary.")
            return []

        result = []
        for addon_id, addon_data in addons.items():
            if not isinstance(addon_data, dict):
                logger.warning(
                    f"Registry entry for '{addon_id}' is not a dict."
                )
                continue
            try:
                meta = AddonMetadata.from_registry_entry(addon_id, addon_data)
                result.append(meta)
            except Exception as e:
                logger.warning(
                    f"Failed to parse registry entry '{addon_id}': {e}"
                )
        return result

    def fetch_registry(self) -> List[AddonMetadata]:
        """
        Fetches and parses the addon registry from the remote repository.
        Returns a list of AddonMetadata objects.
        """
        if yaml is None:
            logger.error("PyYAML is required to fetch the registry.")
            return []

        try:
            logger.info(f"Fetching registry from {ADDON_REGISTRY_URL}")
            with urllib.request.urlopen(
                ADDON_REGISTRY_URL, timeout=10
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

        result: List[AddonMetadata] = []
        if isinstance(parsed, list):
            for addon_data in parsed:
                addon_id = addon_data.get("name")
                if not addon_id:
                    logger.warning(
                        f"Skipping list-based registry entry without a "
                        f"'name': {addon_data}",
                    )
                    continue
                try:
                    meta = AddonMetadata.from_registry_entry(
                        addon_id, addon_data
                    )
                    result.append(meta)
                except Exception as e:
                    logger.warning(
                        "Failed to parse list-based registry entry '%s': %s",
                        addon_id,
                        e,
                    )
            return result

        if isinstance(parsed, dict):
            return self._parse_registry_dict(parsed)

        logger.warning(
            "Registry format is not a recognized list or dictionary."
        )
        return []

    def get_installed_addon(self, addon_id: str) -> Optional[Addon]:
        """
        Finds an installed addon by its canonical ID.

        Returns:
            The Addon object if found, otherwise None.
        """
        return (
            self.loaded_addons.get(addon_id)
            or self.disabled_addons.get(addon_id)
            or self.incompatible_addons.get(addon_id)
            or self.license_required_addons.get(addon_id)
        )

    def check_update_status(
        self, remote_meta: AddonMetadata
    ) -> Tuple[UpdateStatus, Optional[str]]:
        """
        Checks a remote addon against local installations.

        Returns:
            A tuple of (UpdateStatus, local_version_str).
        """
        installed_addon = self.get_installed_addon(remote_meta.name)
        if not installed_addon:
            return (UpdateStatus.NOT_INSTALLED, None)

        local_version = installed_addon.metadata.version
        if local_version is UnknownVersion:
            return (UpdateStatus.UP_TO_DATE, None)

        local_version_str: Optional[str] = str(local_version)
        remote_version = remote_meta.version
        if remote_version is UnknownVersion:
            return (UpdateStatus.UP_TO_DATE, local_version_str)

        is_newer = is_newer_version(str(remote_version), str(local_version))

        if is_newer:
            return (UpdateStatus.UPDATE_AVAILABLE, local_version_str)
        return (UpdateStatus.UP_TO_DATE, local_version_str)

    def check_for_updates(self) -> List[Tuple[Addon, AddonMetadata]]:
        """
        Compares all installed addons against the remote registry to find
        available updates.

        Returns:
            A list of tuples, where each tuple contains the locally installed
            Addon object and the remote AddonMetadata for the update.
        """
        logger.info("Checking for available addon updates...")
        try:
            remote_addons_list = self.fetch_registry()
            if not remote_addons_list:
                logger.warning(
                    "Could not fetch remote registry for update check."
                )
                return []
        except Exception as e:
            logger.error(f"Failed to fetch registry for update check: {e}")
            return []

        remote_addons = {addon.name: addon for addon in remote_addons_list}
        updates_available: List[Tuple[Addon, AddonMetadata]] = []

        all_installed = list(self.loaded_addons.values()) + list(
            self.disabled_addons.values()
        )
        for installed_addon in all_installed:
            remote_meta = remote_addons.get(installed_addon.metadata.name)
            if not remote_meta:
                continue

            local_ver = installed_addon.metadata.version
            remote_ver = remote_meta.version
            if local_ver is UnknownVersion or remote_ver is UnknownVersion:
                continue

            if is_newer_version(str(remote_ver), str(local_ver)):
                logger.info(
                    f"Update found for '{installed_addon.metadata.name}': "
                    f"{local_ver} -> {remote_ver}"
                )
                updates_available.append((installed_addon, remote_meta))

        if not updates_available:
            logger.info("All installed addons are up to date.")

        return updates_available

    def load_installed_addons(self, backend_only: bool = False):
        """
        Scans the addon directories and loads valid addons.

        Args:
            backend_only: If True, only load backend entry points (skip
                frontend/widgets to avoid pulling in GTK dependencies).
                Used by worker processes.
        """
        for addon_dir in self.addon_dirs:
            if not addon_dir.exists():
                if addon_dir == self.install_dir:
                    addon_dir.mkdir(parents=True, exist_ok=True)
                continue

            logger.info(f"Scanning for addons in {addon_dir}...")
            for child in addon_dir.iterdir():
                if child.is_dir():
                    self.load_addon(child.resolve(), backend_only=backend_only)

        # Build manifest at the end of the batch load
        self._build_and_update_manifest()

    def load_addon(self, addon_path: Path, backend_only: bool = False):
        """
        Loads a single addon from a directory.

        Args:
            addon_path: Path to the addon directory.
            backend_only: If True, only load backend entry points (skip
                frontend/widgets to avoid pulling in GTK dependencies).
                Used by worker processes.
        """
        try:
            # 1. Load addon structure without resolving version yet to get
            # canonical name
            addon = Addon.load_from_directory(
                addon_path, version=UnknownVersion
            )
            addon_name = addon.metadata.name

            # 2. Resolve version using the proper addon_name
            is_builtin = not addon_path.is_relative_to(self.install_dir)
            if is_builtin:
                resolved_version = UnknownVersion
            else:
                resolved_version = None
                if self.addon_config:
                    resolved_version = self.addon_config.get_version(
                        addon_name
                    )

                if resolved_version is None:
                    try:
                        resolved_version = get_git_tag_version(addon_path)
                    except RuntimeError:
                        logger.warning(
                            f"No stored version for addon '{addon_name}' "
                            f"at {addon_path} and no git tags found, using "
                            "UnknownVersion"
                        )
                        resolved_version = UnknownVersion

            addon.metadata.version = resolved_version

            # 3. Now completely validate the populated addon
            addon.validate()

            has_backend = addon.metadata.provides.backend is not None
            has_frontend = addon.metadata.provides.frontend is not None

            if not has_backend and not has_frontend:
                self.loaded_addons[addon_name] = addon
                logger.info(f"Loaded asset addon: {addon_name}")
                return

            if self.addon_config:
                state = self.addon_config.get_state(addon_name)
                if state == ConfigAddonState.DISABLED:
                    logger.info(
                        f"Addon '{addon_name}' is disabled, skipping load"
                    )
                    self.disabled_addons[addon_name] = addon
                    return

            if (
                self._check_version_compatibility(addon)
                != UpdateStatus.UP_TO_DATE
            ):
                logger.warning(
                    f"Addon '{addon_name}' is incompatible with "
                    "this version of Rayforge"
                )
                self.incompatible_addons[addon_name] = addon
                return

            allowed, message, purchase_url = self._check_license(addon)
            if not allowed:
                logger.info(
                    f"Addon '{addon_name}' requires license: {message}"
                )
                addon.license_message = message
                addon.purchase_url = purchase_url
                self.license_required_addons[addon_name] = addon
                return

            self.compile_translations(addon_path)

            self._import_and_register(addon, addon.metadata.provides.backend)
            if not backend_only:
                self._import_and_register(
                    addon, addon.metadata.provides.frontend
                )

            version_str = (
                "(builtin)"
                if addon.metadata.version is UnknownVersion
                else str(addon.metadata.version)
            )
            logger.info(f"Loaded addon: {addon_name} {version_str}")

        except (AddonValidationError, FileNotFoundError) as e:
            logger.warning(f"Skipping invalid addon at {addon_path}: {e}")
        except Exception as e:
            logger.error(
                f"Failed to load addon at {addon_path}: {e}",
                exc_info=True,
            )

    def _check_version_compatibility(self, addon: Addon):
        """
        Checks if addon's dependencies are compatible.
        Returns UpdateStatus.UP_TO_DATE if compatible, INCOMPATIBLE otherwise.
        """
        current_version = __version__
        if not current_version:
            logger.warning("Could not determine current rayforge version")
            return UpdateStatus.UP_TO_DATE

        if check_rayforge_compatibility(
            addon.metadata.depends, current_version
        ):
            return UpdateStatus.UP_TO_DATE
        return UpdateStatus.INCOMPATIBLE

    def _check_license(self, addon: Addon) -> Tuple[bool, str, str]:
        """
        Check if addon requires and has valid license.

        Returns:
            Tuple of (is_allowed, message, purchase_url)
        """
        if not self.license_validator:
            return True, "", ""

        license_config = addon.metadata.license

        if not license_config or not license_config.required:
            return True, "", ""

        return self.license_validator.check_license(
            addon.metadata.name, license_config.to_dict()
        )

    def _import_and_register(self, addon: Addon, entry_point: Optional[str]):
        """
        Imports the module specified by entry_point and registers it.

        Args:
            addon: The addon to load.
            entry_point: Entry point string like 'module.submodule',
                         or None to skip.
        """
        if not entry_point:
            return

        name = addon.metadata.name

        # Module name logic: "rayforge_addons.<ADDON_NAME>.<ENTRY_POINT>"
        # Example:
        # 1. rayforge_addons
        # 2. laser_essentials (addon name)
        # 3. laser_essentials.backend (inner python structure)
        module_name = f"rayforge_addons.{name}.{entry_point}"

        module_path = self._resolve_entry_point_path(
            entry_point, addon.root_path
        )
        if not module_path:
            error_msg = f"Entry point {entry_point} not found for {name}."
            logger.error(error_msg)
            self._load_errors[name] = error_msg
            return

        try:
            self._ensure_parent_modules(
                module_name, addon.root_path, entry_point
            )

            spec = importlib.util.spec_from_file_location(
                module_name, module_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                self.plugin_mgr.register(module)
                self.loaded_addons[name] = addon
                if name in self._load_errors:
                    del self._load_errors[name]
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error importing addon {name}: {e}")
            self._load_errors[name] = error_msg

    def _ensure_parent_modules(
        self, module_name: str, root_path: Path, entry_point: str
    ):
        """
        Ensure parent modules exist in sys.modules for relative imports.

        The module structure is:
        rayforge_addons.<ADDON_ID>.<ENTRY_POINT_PARTS...>

        We need to ensure all intermediates exist.
        """
        import types

        # 1. Ensure root 'rayforge_addons'
        if "rayforge_addons" not in sys.modules:
            ns = types.ModuleType("rayforge_addons")
            ns.__path__ = []
            ns.__package__ = "rayforge_addons"
            sys.modules["rayforge_addons"] = ns

        parts = module_name.split(".")
        # parts[0] is rayforge_addons
        # parts[1] is ADDON_ID (e.g. laser_essentials)
        # parts[2:] are the rest

        if len(parts) > 1:
            addon_id = parts[1]
            addon_ns = f"rayforge_addons.{addon_id}"

            # 2. Ensure addon namespace
            if addon_ns not in sys.modules:
                ns = types.ModuleType(addon_ns)
                ns.__path__ = [str(root_path)]
                ns.__package__ = addon_ns
                sys.modules[addon_ns] = ns

            current_ns = addon_ns
            current_path = root_path

            # 3. Walk down entry point parts
            # entry_point = "pkg.sub.mod"
            #   -> parts are [rayforge_addons, ID, pkg, sub, mod]
            # inner_parts are [pkg, sub, mod]
            inner_parts = parts[2:]

            # Iterate up to the second to last part (creating parent packages)
            for i in range(len(inner_parts) - 1):
                pkg_name = inner_parts[i]
                full_pkg_name = f"{current_ns}.{pkg_name}"
                current_path = current_path / pkg_name

                if full_pkg_name not in sys.modules:
                    init_file = current_path / "__init__.py"
                    if init_file.exists():
                        spec = importlib.util.spec_from_file_location(
                            full_pkg_name, init_file
                        )
                        if spec and spec.loader:
                            pkg_module = importlib.util.module_from_spec(spec)
                            sys.modules[full_pkg_name] = pkg_module
                            spec.loader.exec_module(pkg_module)
                    else:
                        ns = types.ModuleType(full_pkg_name)
                        ns.__path__ = [str(current_path)]
                        ns.__package__ = full_pkg_name
                        sys.modules[full_pkg_name] = ns

                current_ns = full_pkg_name

    def _resolve_entry_point_path(
        self, entry_point: str, root_path: Path
    ) -> Optional[Path]:
        """
        Resolve a module path to a file path.

        Args:
            entry_point: Module path like 'my_addon.plugin'
            root_path: The addon root directory.

        Returns:
            Path to the module file, or None if not found.
        """
        module_path = root_path / entry_point.replace(".", "/")
        if module_path.is_dir():
            module_path = module_path / "__init__.py"
        else:
            module_path = module_path.with_suffix(".py")

        if not module_path.exists():
            return None
        return module_path

    def compile_translations(
        self, addon_path: Path, force: bool = False
    ) -> int:
        """
        Compile .po files to .mo files in an addon's locales directory.

        This is called automatically when an addon is installed or loaded.
        It finds all .po files under <addon>/locales/ and compiles them to .mo
        files in the corresponding LC_MESSAGES directories. By default, it only
        compiles if the .mo file is missing or outdated.

        Args:
            addon_path: Path to the installed addon directory.
            force: If True, always compile even if .mo exists and is
                up to date.

        Returns:
            The number of .mo files compiled.
        """
        locales_dir = addon_path / "locales"
        if not locales_dir.exists():
            return 0

        compiled_count = 0
        for po_file in locales_dir.rglob("*.po"):
            mo_file = po_file.with_suffix(".mo")
            if force or needs_compilation(po_file, mo_file):
                if compile_po_to_mo(po_file, mo_file):
                    compiled_count += 1
                    logger.debug(f"Compiled {po_file} -> {mo_file}")

        if compiled_count > 0:
            logger.info(f"Compiled {compiled_count} translation file(s)")
        return compiled_count

    def install_addon(
        self, git_url: str, addon_id: Optional[str] = None
    ) -> Optional[Path]:
        """
        Install an addon from a remote Git repository.

        Args:
            git_url (str): The URL of the repository to clone.
            addon_id (Optional[str]): The canonical ID for the addon,
                provided by the registry. If None, it's derived from the URL
                (for manual installs).
        """
        try:
            importlib.import_module("git")
        except ImportError:
            logger.error("GitPython is required for addon installation.")
            return None

        from git import Repo

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Cloning {git_url} to staging area...")

            try:
                Repo.clone_from(git_url, temp_path)
            except Exception as e:
                logger.error(f"Git clone failed: {e}")
                return None

            try:
                version = get_git_tag_version(temp_path)
            except RuntimeError as e:
                logger.error(f"Failed to get addon version: {e}")
                return None

            try:
                logger.info("Validating addon structure and code safety...")
                addon = Addon.load_from_directory(temp_path, version=version)
                addon.validate()
                logger.info("Validation passed.")

                addon_name = addon.metadata.name
                install_dir_name = addon_id or self._extract_repo_name(git_url)
                final_path = self.install_dir / install_dir_name

                if final_path.exists():
                    logger.info(f"Upgrading existing addon at {final_path}")
                    # Attempt uninstall by true ID
                    self.uninstall_addon(addon_name)

                shutil.copytree(temp_path, final_path, dirs_exist_ok=True)

                self.compile_translations(final_path, force=True)

                if self.addon_config:
                    self.addon_config.set_version(addon_name, version)

                logger.info(f"Successfully installed addon to {final_path}")
                self.load_addon(final_path)

                self._build_and_update_manifest()
                self._restart_workers()
                return final_path

            except AddonValidationError as e:
                logger.error(f"Addon validation failed: {e}")
                return None
            except Exception as e:
                logger.error(f"Installation failed: {e}", exc_info=True)
                return None

    def uninstall_addon(self, addon_name: str) -> bool:
        """
        Deletes the addon directory and unloads the module.
        """
        addon = (
            self.loaded_addons.get(addon_name)
            or self.incompatible_addons.get(addon_name)
            or self.disabled_addons.get(addon_name)
            or self.license_required_addons.get(addon_name)
        )
        if not addon:
            logger.warning(
                f"Attempted to uninstall unknown or already "
                f"uninstalled addon: {addon_name}"
            )
            # Fallback: scan install_dir for matching addon
            for child in self.install_dir.iterdir():
                if child.is_dir() and (child / "rayforge-addon.yaml").exists():
                    try:
                        temp_addon = Addon.load_from_directory(
                            child, version=UnknownVersion
                        )
                        if temp_addon.metadata.name == addon_name:
                            self._cleanup_directory(child)
                            return True
                    except Exception as e:
                        logger.error(
                            f"Failed to uninstall addon {addon_name}: {e}"
                        )
            return False

        addon_path = addon.root_path

        try:
            if addon_path.exists() and addon_path.is_dir():
                self._cleanup_directory(addon_path)
                logger.info(f"Uninstalled addon at {addon_path}")

            # Robust unloading: Find all modules loaded from this addon's
            # directory
            # This handles any module structure (old style or new style)
            addon_path_str = str(addon_path.resolve())
            modules_to_unload = []
            for name, module in list(sys.modules.items()):
                if not hasattr(module, "__file__") or not module.__file__:
                    continue
                try:
                    # Resolve to absolute path to match
                    mod_path = str(Path(module.__file__).resolve())
                    if mod_path.startswith(addon_path_str):
                        modules_to_unload.append(name)
                except Exception:
                    logger.warning(
                        f"Could not resolve path for module {name}, "
                        "skipping unload."
                    )

            for module_name in modules_to_unload:
                module = sys.modules.get(module_name)
                if module:
                    try:
                        self.plugin_mgr.unregister(module)
                    except ValueError:
                        pass
                    del sys.modules[module_name]
                    logger.info(f"Unloaded module: {module_name}")

            if addon_name in self.loaded_addons:
                del self.loaded_addons[addon_name]
            if addon_name in self.incompatible_addons:
                del self.incompatible_addons[addon_name]
            if addon_name in self.disabled_addons:
                del self.disabled_addons[addon_name]
            if addon_name in self.license_required_addons:
                del self.license_required_addons[addon_name]
            if addon_name in self._load_errors:
                del self._load_errors[addon_name]
            self._pending_unloads.discard(addon_name)

            if self.addon_config:
                self.addon_config.remove_state(addon_name)

            self._build_and_update_manifest()
            self._restart_workers()
            return True

        except Exception as e:
            logger.error(f"Failed to uninstall {addon_name}: {e}")
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

    def _cleanup_directory(self, addon_path: Path):
        """
        Clean up a directory.
        """
        try:
            if addon_path.exists():
                shutil.rmtree(addon_path)
                logger.debug(f"Cleaned up directory: {addon_path}")
        except Exception as e:
            logger.error(f"Failed to clean up {addon_path}: {e}")

    def enable_addon(self, addon_name: str) -> bool:
        """
        Enable an addon. Returns True if successful.

        The addon will be loaded on the next application start or when
        load_addon is called explicitly.
        """
        if not self.addon_config:
            logger.warning("Cannot enable addon: addon_config not configured")
            return False

        addon = self.disabled_addons.get(addon_name)
        if not addon:
            logger.warning(f"Cannot enable addon: '{addon_name}' not found")
            return False

        self.addon_config.set_state(addon_name, ConfigAddonState.ENABLED)
        del self.disabled_addons[addon_name]

        if self._check_version_compatibility(addon) != UpdateStatus.UP_TO_DATE:
            self.incompatible_addons[addon_name] = addon
            logger.info(f"Addon '{addon_name}' enabled but is incompatible")
        else:
            self._import_and_register(addon, addon.metadata.provides.backend)
            self._import_and_register(addon, addon.metadata.provides.frontend)
            self._call_registration_hooks()
            logger.info(f"Addon '{addon_name}' enabled and loaded")

        self._build_and_update_manifest()
        self._restart_workers()
        return True

    def disable_addon(self, addon_name: str) -> bool:
        """
        Disable an addon. Returns True if immediate, False if deferred.

        If jobs are active, the addon is marked for deferred unload and
        will be unloaded when complete_pending_unloads() is called.
        """
        if not self.addon_config:
            logger.warning("Cannot disable addon: addon_config not configured")
            return False

        addon = self.loaded_addons.get(addon_name)
        if not addon:
            logger.warning(f"Cannot disable addon: '{addon_name}' not loaded")
            return False

        if self.is_job_active_callback and self.is_job_active_callback():
            logger.info(
                f"Jobs active, deferring unload of addon '{addon_name}'"
            )
            self._pending_unloads.add(addon_name)
            self.addon_config.set_state(addon_name, ConfigAddonState.DISABLED)
            return False

        self._do_unload_addon(addon_name, addon)
        self._build_and_update_manifest()
        self._restart_workers()
        return True

    def _do_unload_addon(self, addon_name: str, addon: Addon):
        """Perform the actual unload of an addon."""
        if self.addon_config:
            self.addon_config.set_state(addon_name, ConfigAddonState.DISABLED)

        # Robust unloading: Find all modules loaded from this addon's directory
        # This handles any module structure (old style or new style)
        addon_path_str = str(addon.root_path.resolve())
        modules_to_unload = []
        for name, module in list(sys.modules.items()):
            if not hasattr(module, "__file__") or not module.__file__:
                continue
            try:
                mod_path = str(Path(module.__file__).resolve())
                if mod_path.startswith(addon_path_str):
                    modules_to_unload.append(name)
            except Exception:
                logger.warning(
                    f"Could not resolve path for module {name}, "
                    "skipping unload."
                )

        self.plugin_mgr.hook.on_unload()

        for module_name in modules_to_unload:
            module = sys.modules.get(module_name)
            if module:
                try:
                    registered = self.plugin_mgr.get_plugin(module_name)
                    if registered is not None:
                        self.plugin_mgr.unregister(registered)
                except (TypeError, AttributeError, ValueError):
                    pass
                del sys.modules[module_name]
                logger.debug(f"Unloaded module: {module_name}")

        self._unregister_addon_items(addon_name)

        del self.loaded_addons[addon_name]
        self.disabled_addons[addon_name] = addon
        self._pending_unloads.discard(addon_name)
        logger.info(f"Addon '{addon_name}' disabled")

    def _unregister_addon_items(self, addon_name: str):
        """Unregister all items registered by an addon."""
        for name, registry in self.registries.items():
            count = registry.unregister_all_from_addon(addon_name)
            if count:
                logger.debug(
                    f"Unregistered {count} items from {name} for {addon_name}"
                )

    def _call_registration_hooks(self):
        """Call registration hooks for newly loaded addons."""
        step_registry = self.registries.get("step_registry")
        producer_registry = self.registries.get("producer_registry")
        widget_registry = self.registries.get("widget_registry")
        action_registry = self.registries.get("action_registry")
        layout_registry = self.registries.get("layout_registry")

        if step_registry:
            self.plugin_mgr.hook.register_steps(step_registry=step_registry)
        if producer_registry:
            self.plugin_mgr.hook.register_producers(
                producer_registry=producer_registry
            )
        if widget_registry:
            self.plugin_mgr.hook.register_step_widgets(
                widget_registry=widget_registry
            )
        if layout_registry:
            self.plugin_mgr.hook.register_layout_strategies(
                layout_registry=layout_registry
            )
        if action_registry:
            self.plugin_mgr.hook.register_actions(
                action_registry=action_registry
            )

    def complete_pending_unloads(self) -> List[str]:
        """
        Complete any pending addon unloads.

        Should be called when jobs finish to unload addons that were
        disabled while jobs were active.

        Returns:
            List of addon names that were unloaded.
        """
        if not self._pending_unloads:
            return []

        unloaded = []
        for addon_name in list(self._pending_unloads):
            addon = self.loaded_addons.get(addon_name)
            if addon:
                self._do_unload_addon(addon_name, addon)
                unloaded.append(addon_name)

        if unloaded:
            self._build_and_update_manifest()
            self._restart_workers()

        return unloaded

    def has_pending_unloads(self) -> bool:
        """Check if there are addons waiting to be unloaded."""
        return len(self._pending_unloads) > 0

    def get_pending_unloads(self) -> Set[str]:
        """Get the set of addon names pending unload."""
        return self._pending_unloads.copy()

    def is_addon_enabled(self, addon_name: str) -> bool:
        """Check if an addon is currently enabled and loaded."""
        return addon_name in self.loaded_addons

    def get_addon_state(self, addon_name: str) -> str:
        """
        Get the current state of an addon.

        Returns one of: 'enabled', 'disabled', 'pending_unload',
        'load_error', 'incompatible', 'license_required', 'not_installed'
        """
        if addon_name in self._pending_unloads:
            return AddonState.PENDING_UNLOAD.value
        if addon_name in self._load_errors:
            return AddonState.LOAD_ERROR.value
        if addon_name in self.loaded_addons:
            return AddonState.ENABLED.value
        if addon_name in self.disabled_addons:
            return AddonState.DISABLED.value
        if addon_name in self.incompatible_addons:
            return AddonState.INCOMPATIBLE.value
        if addon_name in self.license_required_addons:
            return AddonState.LICENSE_REQUIRED.value
        return AddonState.NOT_INSTALLED.value

    def get_addon_error(self, addon_name: str) -> Optional[str]:
        """Get the error message for an addon that failed to load."""
        return self._load_errors.get(addon_name)

    def reload_addon(self, addon_name: str) -> bool:
        """
        Reload an addon (disable then enable).

        Returns True if successful.
        """
        if addon_name not in self.loaded_addons:
            logger.warning(
                f"Cannot reload addon '{addon_name}': not currently loaded"
            )
            return False

        if self.is_job_active_callback and self.is_job_active_callback():
            logger.warning(
                f"Cannot reload addon '{addon_name}': jobs are active"
            )
            return False

        addon = self.loaded_addons.get(addon_name)
        if not addon:
            return False

        self._do_unload_addon(addon_name, addon)

        del self.disabled_addons[addon_name]
        if self.addon_config:
            self.addon_config.set_state(addon_name, ConfigAddonState.ENABLED)

        if self._check_version_compatibility(addon) != UpdateStatus.UP_TO_DATE:
            self.incompatible_addons[addon_name] = addon
            logger.info(f"Addon '{addon_name}' reloaded but is incompatible")
            return False

        self._import_and_register(addon, addon.metadata.provides.backend)
        self._import_and_register(addon, addon.metadata.provides.frontend)
        self._call_registration_hooks()
        if addon_name in self.loaded_addons:
            logger.info(f"Addon '{addon_name}' reloaded successfully")
            self.addon_reloaded.send(self, addon_name=addon_name)
            self._build_and_update_manifest()
            self._restart_workers()
            return True
        else:
            logger.error(f"Failed to reload addon '{addon_name}'")
            return False

    def _find_dependents(self, addon_name: str) -> List[str]:
        """
        Find all enabled addons that depend on the given addon.

        Returns:
            List of addon names that depend on this addon.
        """
        dependents = []
        for name, addon in self.loaded_addons.items():
            for req in addon.metadata.requires:
                req_name, _ = parse_requirement(req)
                if req_name == addon_name:
                    dependents.append(name)
                    break
        return dependents

    def can_disable(self, addon_name: str) -> Tuple[bool, str]:
        """
        Check if an addon can be disabled.

        Returns:
            Tuple of (can_disable, reason). If can_disable is False,
            reason contains the explanation.
        """
        dependents = self._find_dependents(addon_name)
        if dependents:
            return False, f"Required by: {', '.join(dependents)}"
        return True, ""

    def get_missing_dependencies(
        self, addon_name: str
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Get missing or disabled dependencies for an addon.

        Returns:
            List of (name, version_spec) tuples for missing dependencies.
        """
        addon = (
            self.loaded_addons.get(addon_name)
            or self.disabled_addons.get(addon_name)
            or self.incompatible_addons.get(addon_name)
        )
        if not addon:
            return []

        missing = []
        for req in addon.metadata.requires:
            req_name, version_spec = parse_requirement(req)
            if req_name not in self.loaded_addons:
                missing.append((req_name, version_spec))
        return missing

    def enable_addon_with_deps(
        self, addon_name: str
    ) -> Tuple[bool, List[str]]:
        """
        Enable an addon along with its missing dependencies.

        Returns:
            Tuple of (success, list_of_enabled_addons).
        """
        addon = self.disabled_addons.get(addon_name)
        if not addon:
            logger.warning(f"Cannot enable addon: '{addon_name}' not found")
            return False, []

        missing = self.get_missing_dependencies(addon_name)
        enabled = []

        for req_name, _ in missing:
            if req_name in self.disabled_addons:
                if not self.enable_addon(req_name):
                    logger.error(
                        f"Failed to enable dependency '{req_name}' "
                        f"for '{addon_name}'"
                    )
                    for name in enabled:
                        self.disable_addon(name)
                    return False, []
                enabled.append(req_name)
            elif req_name not in self.loaded_addons:
                logger.error(
                    f"Missing dependency '{req_name}' for '{addon_name}' "
                    "is not installed"
                )
                for name in enabled:
                    self.disable_addon(name)
                return False, []

        if not self.enable_addon(addon_name):
            for name in enabled:
                self.disable_addon(name)
            return False, []

        enabled.append(addon_name)
        return True, enabled

    def get_license_required_addon(self, addon_name: str) -> Optional[Addon]:
        """
        Get an addon that requires a license.

        Returns:
            The Addon object if found in license_required_addons, else None.
        """
        return self.license_required_addons.get(addon_name)

    def recheck_license(self, addon_name: str) -> Tuple[bool, str]:
        """
        Recheck the license for an addon and attempt to load it if valid.

        Returns:
            Tuple of (success, message).
        """
        addon = self.license_required_addons.get(addon_name)
        if not addon:
            return False, "Addon not in license-required state"

        allowed, message, _ = self._check_license(addon)
        if not allowed:
            return False, message

        del self.license_required_addons[addon_name]

        self._import_and_register(addon, addon.metadata.provides.backend)
        self._import_and_register(addon, addon.metadata.provides.frontend)

        if addon_name in self.loaded_addons:
            self._build_and_update_manifest()
            self._restart_workers()
            return True, "License validated, addon loaded successfully"

        return False, "Failed to load addon after license validation"

    def get_all_license_required_addons(self) -> Dict[str, Addon]:
        """
        Get all addons that require a license.

        Returns:
            Dict of addon_name -> Addon for all license-required addons.
        """
        return dict(self.license_required_addons)
