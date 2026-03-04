import importlib.abc
import importlib.machinery
import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


class AddonModuleFinder(importlib.abc.MetaPathFinder):
    """
    Lazy module finder for rayforge_addons.* modules.

    This finder is installed as the first entry in sys.meta_path
    to handle addon module imports before the addon is loaded.
    """

    def __init__(
        self,
        addon_modules: Optional[Dict[str, str]] = None,
        shared_dict: Optional[dict] = None,
    ):
        """
        Initialize the lazy module finder.

        Args:
            addon_modules: Mapping from full module name to file path.
            shared_dict: A multiprocessing Manager dict to look up paths
                      dynamically.
        """
        self._addon_modules = addon_modules
        self._shared_dict = shared_dict

    @property
    def addon_modules(self):
        """Get addon modules, checking shared dict first."""
        if self._shared_dict:
            paths = self._shared_dict.get("addon_module_paths", {})
            if paths:
                return paths
        return self._addon_modules

    def find_spec(self, fullname: str, path, target=None):
        if fullname != "rayforge_addons" and not fullname.startswith(
            "rayforge_addons."
        ):
            return None

        if fullname in sys.modules:
            return None

        modules = self.addon_modules
        if modules is None:
            return None

        if fullname in modules:
            module_path = Path(modules[fullname])
            if module_path.exists():
                spec = importlib.util.spec_from_file_location(
                    fullname, module_path
                )
                return spec

        return None


_installed_finder: Optional["AddonModuleFinder"] = None


def install_addon_finder(
    addon_modules: Optional[Dict[str, str]] = None,
    shared_dict: Optional[dict] = None,
):
    """
    Install the lazy addon module finder.

    This should be called in worker processes before unpickling tasks
    to handle addon module imports before the addon is loaded.

    Args:
        addon_modules: Mapping from full module name to file path.
        shared_dict: A multiprocessing Manager dict to look up paths
                      dynamically.
    """
    global _installed_finder

    if _installed_finder is not None:
        return

    _installed_finder = AddonModuleFinder(
        addon_modules=addon_modules, shared_dict=shared_dict
    )
    sys.meta_path.insert(0, _installed_finder)
    modules_count = len(addon_modules) if addon_modules else 0
    logger.debug(f"Installed AddonModuleFinder with {modules_count} modules")


def collect_module_paths(addon_dirs: List[Path]) -> Dict[str, str]:
    """
    Scan addon directories and return module name -> file path mapping.

    This function scans the given addon directories and collects all Python
    module paths, using the Addon class to properly parse metadata.

    Args:
        addon_dirs: List of directories to scan for addons.

    Returns:
        Dictionary mapping fully qualified module names to file paths.
        E.g., {"rayforge_addons.my_addon.backend": "/path/to/backend.py"}
    """
    from rayforge.shared.util.versioning import UnknownVersion

    from .addon import Addon

    paths: Dict[str, str] = {}

    for addon_dir in addon_dirs:
        if not addon_dir.exists():
            continue

        for child in addon_dir.iterdir():
            if not child.is_dir():
                continue

            try:
                addon = Addon.load_from_directory(
                    child, version=UnknownVersion
                )
                name = addon.metadata.name

                backend_ep = addon.metadata.provides.backend
                frontend_ep = addon.metadata.provides.frontend

                for entry_point in [backend_ep, frontend_ep]:
                    if not entry_point:
                        continue

                    module_name = f"rayforge_addons.{name}.{entry_point}"
                    module_path = _resolve_entry_point_path(
                        entry_point, addon.root_path
                    )
                    if module_path:
                        paths[module_name] = str(module_path)

                for py_file in addon.root_path.rglob("*.py"):
                    rel_path = py_file.relative_to(addon.root_path)

                    if rel_path.name == "__init__.py":
                        parts = rel_path.parts[:-1]
                        if parts:
                            module_name = (
                                f"rayforge_addons.{name}.{'.'.join(parts)}"
                            )
                            paths[module_name] = str(py_file)
                    else:
                        module_path_str = str(rel_path.with_suffix(""))
                        module_name = (
                            f"rayforge_addons.{name}."
                            f"{module_path_str.replace('/', '.')}"
                        )
                        paths[module_name] = str(py_file)
            except Exception:
                pass

    return paths


def _resolve_entry_point_path(
    entry_point: str, root_path: Path
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


def ensure_addon_namespaces(addon_dirs: List[Path]) -> None:
    """
    Create rayforge_addons.* namespace packages in sys.modules.

    This must be called before unpickling to prevent ModuleNotFoundError
    when unpickling references to rayforge_addons.* modules.

    The unpickler doesn't use the import system, so namespaces must be
    in sys.modules before unpickling happens.

    Args:
        addon_dirs: List of directories to scan for addons.
    """
    from rayforge.shared.util.versioning import UnknownVersion

    from .addon import Addon

    if "rayforge_addons" not in sys.modules:
        ns = types.ModuleType("rayforge_addons")
        ns.__path__ = []
        ns.__package__ = "rayforge_addons"
        sys.modules["rayforge_addons"] = ns
        spec = importlib.machinery.ModuleSpec(
            "rayforge_addons", loader=None, is_package=True
        )
        spec.submodule_search_locations = []
        ns.__spec__ = spec

    for addon_dir in addon_dirs:
        if not addon_dir.exists():
            continue

        for child in addon_dir.iterdir():
            if not child.is_dir():
                continue

            try:
                addon = Addon.load_from_directory(
                    child, version=UnknownVersion
                )
                name = addon.metadata.name

                ns_name = f"rayforge_addons.{name}"
                if ns_name not in sys.modules:
                    ns = types.ModuleType(ns_name)
                    ns.__path__ = []
                    ns.__package__ = ns_name
                    sys.modules[ns_name] = ns
                    spec = importlib.machinery.ModuleSpec(
                        ns_name, loader=None, is_package=True
                    )
                    spec.submodule_search_locations = []
                    ns.__spec__ = spec

                backend_ep = addon.metadata.provides.backend
                frontend_ep = addon.metadata.provides.frontend

                for entry_point in [backend_ep, frontend_ep]:
                    if not entry_point:
                        continue

                    entry_parts = entry_point.split(".")
                    for i in range(1, len(entry_parts) + 1):
                        parts_prefix = ".".join(entry_parts[:i])
                        int_ns_name = f"rayforge_addons.{name}.{parts_prefix}"
                        if int_ns_name not in sys.modules:
                            ns = types.ModuleType(int_ns_name)
                            ns.__path__ = []
                            ns.__package__ = int_ns_name
                            sys.modules[int_ns_name] = ns
                            int_spec = importlib.machinery.ModuleSpec(
                                int_ns_name, loader=None, is_package=True
                            )
                            int_spec.submodule_search_locations = []
                            ns.__spec__ = int_spec
            except Exception:
                pass
