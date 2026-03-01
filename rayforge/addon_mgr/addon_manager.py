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
from ..shared.util.po_compiler import compile_po_to_mo
from ..shared.util.versioning import (
    check_rayforge_compatibility,
    get_git_tag_version,
    is_newer_version,
    parse_requirement,
    UnknownVersion,
)
from .addon import Addon, AddonMetadata, AddonValidationError

if TYPE_CHECKING:
    pass

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
                'menu_registry'. Each registry must implement the
                AddonRegistry protocol.
        """
        self.addon_dirs = addon_dirs
        self.install_dir = install_dir
        self.plugin_mgr = plugin_mgr
        self.addon_config = addon_config
        self.is_job_active_callback = is_job_active_callback
        self.registries = registries or {}
        self.loaded_addons: Dict[str, Addon] = {}
        self.incompatible_addons: Dict[str, Addon] = {}
        self.disabled_addons: Dict[str, Addon] = {}
        self._pending_unloads: Set[str] = set()
        self._load_errors: Dict[str, str] = {}

        self.addon_reloaded = Signal()

    def set_registries(self, registries: Dict[str, "AddonRegistry"]):
        """
        Set the registries dict for addon cleanup.

        Args:
            registries: Dict mapping hook parameter names to registry
                instances. Expected keys: 'step_registry',
                'producer_registry', 'widget_registry', 'menu_registry'.
        """
        self.registries = registries

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
        return self.loaded_addons.get(addon_id)

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

        for installed_addon in self.loaded_addons.values():
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
            is_builtin = not addon_path.is_relative_to(self.install_dir)
            version = None

            if is_builtin:
                version = UnknownVersion
            else:
                if self.addon_config:
                    stored_version = self.addon_config.get_version(
                        addon_path.name
                    )
                    if stored_version is not None:
                        version = stored_version
                if version is None:
                    try:
                        version = get_git_tag_version(addon_path)
                    except RuntimeError:
                        logger.warning(
                            f"No stored version for '{addon_path.name}' "
                            "and no git tags found, using UnknownVersion"
                        )
                        version = UnknownVersion

            addon = Addon.load_from_directory(addon_path, version=version)

            has_backend = addon.metadata.provides.backend is not None
            has_frontend = addon.metadata.provides.frontend is not None

            if not has_backend and not has_frontend:
                self.loaded_addons[addon.metadata.name] = addon
                logger.info(f"Loaded asset addon: {addon.metadata.name}")
                return

            if self.addon_config:
                state = self.addon_config.get_state(addon.metadata.name)
                if state == ConfigAddonState.DISABLED:
                    logger.info(
                        f"Addon '{addon.metadata.name}' is disabled, "
                        "skipping load"
                    )
                    self.disabled_addons[addon.metadata.name] = addon
                    return

            if (
                self._check_version_compatibility(addon)
                != UpdateStatus.UP_TO_DATE
            ):
                logger.warning(
                    f"Addon '{addon.metadata.name}' is incompatible with "
                    "this version of Rayforge"
                )
                self.incompatible_addons[addon.metadata.name] = addon
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
                else f"v{addon.metadata.version}"
            )
            logger.info(f"Loaded addon: {addon.metadata.name} {version_str}")

        except (AddonValidationError, FileNotFoundError) as e:
            logger.warning(f"Skipping invalid addon at {addon_path.name}: {e}")
        except Exception as e:
            logger.error(
                f"Failed to load addon {addon_path.name}: {e}",
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
        module_name = f"rayforge_plugins.{name}.{entry_point}"

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
            logger.error(f"Error importing plugin {name}: {e}")
            self._load_errors[name] = error_msg

    def _ensure_parent_modules(
        self, module_name: str, root_path: Path, entry_point: str
    ):
        """
        Ensure parent modules exist in sys.modules for relative imports.

        For entry_point 'laser_essentials.backend', we need:
        - rayforge_plugins (namespace)
        - rayforge_plugins.{name} (namespace)
        - rayforge_plugins.{name}.laser_essentials (actual addon module)
        """
        import types

        parts = module_name.split(".")
        if parts[0] not in sys.modules:
            ns = types.ModuleType(parts[0])
            ns.__path__ = []
            ns.__package__ = parts[0]
            sys.modules[parts[0]] = ns

        if len(parts) > 1:
            parent = f"{parts[0]}.{parts[1]}"
            if parent not in sys.modules:
                ns = types.ModuleType(parent)
                ns.__path__ = []
                ns.__package__ = parent
                sys.modules[parent] = ns

        entry_parts = entry_point.split(".")
        if len(entry_parts) > 1:
            pkg_name = f"{parts[0]}.{parts[1]}.{entry_parts[0]}"
            if pkg_name not in sys.modules:
                pkg_path = root_path / entry_parts[0] / "__init__.py"
                if pkg_path.exists():
                    spec = importlib.util.spec_from_file_location(
                        pkg_name, pkg_path
                    )
                    if spec and spec.loader:
                        pkg_module = importlib.util.module_from_spec(spec)
                        sys.modules[pkg_name] = pkg_module
                        spec.loader.exec_module(pkg_module)
                else:
                    ns = types.ModuleType(pkg_name)
                    ns.__path__ = [str(root_path / entry_parts[0])]
                    ns.__package__ = pkg_name
                    sys.modules[pkg_name] = ns

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

    def compile_translations(self, addon_path: Path) -> int:
        """
        Compile .po files to .mo files in an addon's locales directory.

        This is called automatically when an addon is installed. It finds
        all .po files under <addon>/locales/ and compiles them to .mo
        files in the corresponding LC_MESSAGES directories.

        Args:
            addon_path: Path to the installed addon directory.

        Returns:
            The number of .mo files compiled.
        """
        locales_dir = addon_path / "locales"
        if not locales_dir.exists():
            return 0

        compiled_count = 0
        for po_file in locales_dir.rglob("*.po"):
            mo_file = po_file.with_suffix(".mo")
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

                install_name = addon_id or self._extract_repo_name(git_url)
                final_path = self.install_dir / install_name

                if final_path.exists():
                    logger.info(f"Upgrading existing addon at {final_path}")
                    self.uninstall_addon(install_name)

                shutil.copytree(temp_path, final_path, dirs_exist_ok=True)

                self.compile_translations(final_path)

                if self.addon_config:
                    self.addon_config.set_version(install_name, version)

                logger.info(f"Successfully installed addon to {final_path}")
                self.load_addon(final_path)
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
        )
        if not addon:
            logger.warning(
                f"Attempted to uninstall unknown or already "
                f"uninstalled addon: {addon_name}"
            )
            addon_path = self.install_dir / addon_name
            if addon_path.exists():
                self._cleanup_directory(addon_path)
                return True
            return False

        addon_path = addon.root_path

        try:
            if addon_path.exists() and addon_path.is_dir():
                self._cleanup_directory(addon_path)
                logger.info(f"Uninstalled addon at {addon_path}")

            prefix = f"rayforge_plugins.{addon_name}."
            modules_to_unload = [
                name for name in sys.modules if name.startswith(prefix)
            ]
            base_module = f"rayforge_plugins.{addon_name}"
            if base_module in sys.modules:
                modules_to_unload.append(base_module)

            for module_name in modules_to_unload:
                module = sys.modules.get(module_name)
                if module:
                    self.plugin_mgr.unregister(module)
                    del sys.modules[module_name]
                    logger.info(f"Unloaded module: {module_name}")

            if addon_name in self.loaded_addons:
                del self.loaded_addons[addon_name]
            if addon_name in self.incompatible_addons:
                del self.incompatible_addons[addon_name]
            if addon_name in self.disabled_addons:
                del self.disabled_addons[addon_name]

            if self.addon_config:
                self.addon_config.remove_state(addon_name)

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
        return True

    def _do_unload_addon(self, addon_name: str, addon: Addon):
        """Perform the actual unload of an addon."""
        if self.addon_config:
            self.addon_config.set_state(addon_name, ConfigAddonState.DISABLED)

        prefix = f"rayforge_plugins.{addon_name}."
        modules_to_unload = [
            name for name in list(sys.modules) if name.startswith(prefix)
        ]

        base_module = f"rayforge_plugins.{addon_name}"
        if base_module in sys.modules:
            modules_to_unload.append(base_module)

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
        menu_registry = self.registries.get("menu_registry")

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
        if menu_registry:
            self.plugin_mgr.hook.register_menu_items(
                menu_registry=menu_registry
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
        'load_error', 'incompatible', 'not_installed'
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
