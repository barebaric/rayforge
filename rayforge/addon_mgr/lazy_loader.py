import importlib.abc
import importlib.machinery
import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Optional
from .manifest import AddonManifest


logger = logging.getLogger(__name__)


class AddonModuleFinder(importlib.abc.MetaPathFinder):
    """
    Lazy module finder for rayforge_addons.* modules.

    This finder is installed as the first entry in sys.meta_path
    to handle addon module imports using the pre-computed AddonManifest.
    """

    def __init__(self, shared_dict: dict):
        """
        Initialize the lazy module finder.

        Args:
            shared_dict: A multiprocessing Manager dict containing the
              manifest.
        """
        self._shared_dict = shared_dict

    def find_spec(self, fullname: str, path, target=None):
        if fullname != "rayforge_addons" and not fullname.startswith(
            "rayforge_addons."
        ):
            return None

        if fullname in sys.modules:
            return None

        manifest = self._shared_dict.get("addon_manifest")
        if not manifest:
            return None

        # Check for actual python modules
        if fullname in manifest.module_paths:
            module_path = Path(manifest.module_paths[fullname])
            if module_path.exists():
                return importlib.util.spec_from_file_location(
                    fullname, module_path
                )

        # Check for namespace packages (directories without __init__.py)
        if (
            fullname in manifest.namespaces
            and fullname not in manifest.module_paths
        ):
            spec = importlib.machinery.ModuleSpec(
                fullname, loader=None, is_package=True
            )
            spec.submodule_search_locations = []
            return spec

        return None


_installed_finder: Optional["AddonModuleFinder"] = None


def install_addon_finder(shared_dict: dict):
    """
    Install the lazy addon module finder.

    This should be called in worker processes before unpickling tasks
    to handle addon module imports via the AddonManifest.

    Args:
        shared_dict: A multiprocessing Manager dict containing the manifest.
    """
    global _installed_finder

    if _installed_finder is not None:
        # If finder is already installed, update its shared dict reference.
        # This handles cases where the worker pool is restarted or tests
        # reuse the process.
        _installed_finder._shared_dict = shared_dict
        return

    _installed_finder = AddonModuleFinder(shared_dict=shared_dict)
    sys.meta_path.insert(0, _installed_finder)
    logger.debug("Installed AddonModuleFinder using manifest.")


def reset_addon_finder():
    """
    Remove the installed addon finder. Used for test teardown.
    """
    global _installed_finder
    if _installed_finder is not None:
        if _installed_finder in sys.meta_path:
            sys.meta_path.remove(_installed_finder)
        _installed_finder = None


def ensure_addon_namespaces(manifest: "AddonManifest") -> None:
    """
    Create namespace packages in sys.modules from a manifest.

    This must be called before unpickling to prevent ModuleNotFoundError
    when unpickling references to rayforge_addons.* modules.

    The unpickler doesn't use the import system, so namespaces must be
    in sys.modules before unpickling happens.

    Args:
        manifest: The AddonManifest containing the namespaces to create.
    """
    # Sort for deterministic creation order (parent before child)
    for ns_name in sorted(list(manifest.namespaces)):
        # If the namespace is actually a real package (has an __init__.py),
        # do NOT create a dummy namespace for it. Let the AddonModuleFinder
        # load the real __init__.py when it's imported.
        if ns_name in manifest.module_paths:
            continue

        if ns_name not in sys.modules:
            ns = types.ModuleType(ns_name)
            # __path__ must be set for it to be recognized as a package
            ns.__path__ = []
            ns.__package__ = ns_name
            sys.modules[ns_name] = ns
            spec = importlib.machinery.ModuleSpec(
                ns_name, loader=None, is_package=True
            )
            spec.submodule_search_locations = []
            ns.__spec__ = spec
