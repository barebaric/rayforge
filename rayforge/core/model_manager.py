"""Model manager for 3D model assets in Rayforge."""

import importlib.resources
import logging
from gettext import gettext as _
from pathlib import Path
from typing import Dict, List, Optional

from blinker import Signal

from ..addon_mgr.addon_manager import AddonRegistry
from .model import Model, ModelLibrary

logger = logging.getLogger(__name__)

VALID_MODEL_EXTENSIONS = {".glb", ".gltf"}


class ModelManager(AddonRegistry):
    """
    Application-wide read-only resolver for 3D model assets.

    Maintains an ordered collection of :class:`ModelLibrary`
    instances.  When resolving a model path the libraries are
    searched in registration order, so earlier libraries take
    precedence.

    Libraries are registered by the core bundled resources,
    device profiles, and addons.  There is no user-writable
    library — models travel with device profiles.
    """

    def __init__(self):
        self.changed = Signal()
        self._libraries: Dict[str, ModelLibrary] = {}
        self._library_addons: Dict[str, Optional[str]] = {}

    def get_libraries(self) -> List[ModelLibrary]:
        """Return all registered libraries in registration order."""
        return list(self._libraries.values())

    def add_library(
        self,
        library: ModelLibrary,
        addon_name: Optional[str] = None,
    ) -> bool:
        """
        Register a model library.

        Args:
            library: The library to add.
            addon_name: If provided, tracks ownership so the library
                can be removed when the addon is disabled.

        Returns:
            ``True`` if added, ``False`` if *library_id* already
            exists.
        """
        if library.library_id in self._libraries:
            return False
        self._libraries[library.library_id] = library
        self._library_addons[library.library_id] = addon_name
        return True

    def add_library_from_path(
        self,
        path: Path,
        display_name: Optional[str] = None,
        read_only: bool = True,
        addon_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create and register a library from a directory path.

        Convenience method for addons.

        Args:
            path: Root directory for the library.
            display_name: Human-readable name (defaults to directory
                name).
            read_only: Whether the library is read-only.
            addon_name: Owning addon, if any.

        Returns:
            The *library_id* on success, ``None`` on failure.
        """
        library_id = path.name
        library = ModelLibrary(
            library_id=library_id,
            display_name=display_name or path.name.title(),
            path=path,
            read_only=read_only,
        )
        if self.add_library(library, addon_name=addon_name):
            return library.library_id
        return None

    def register_bundled_library(self):
        """Register the bundled (core) model library if available."""
        core_path = self._get_bundled_path()
        if core_path is None:
            return
        lib = ModelLibrary(
            library_id="core",
            display_name=_("Core"),
            path=core_path,
            read_only=True,
            icon_name="addon-builtin-symbolic",
        )
        self.add_library(lib)

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Remove all libraries registered by the named addon.

        Implements the AddonRegistry protocol.
        """
        to_remove = [
            lid
            for lid, name in self._library_addons.items()
            if name == addon_name
        ]
        for lid in to_remove:
            if lid in self._libraries:
                del self._libraries[lid]
            del self._library_addons[lid]
        if to_remove:
            self.changed.send(self)
        return len(to_remove)

    def resolve(self, model: Model) -> Optional[Path]:
        """
        Resolve a Model to an absolute filesystem path.

        Searches registered libraries in order.  Absolute paths are
        returned directly if the file exists.
        """
        path = model.path

        if path.is_absolute():
            return path if path.is_file() else None

        for lib in self._libraries.values():
            full = lib.path / path
            if full.is_file():
                return full

        return None

    def get_models(self, library: ModelLibrary) -> List[Model]:
        """
        List model files directly in the library root directory.

        Args:
            library: The ModelLibrary to query.

        Returns:
            Sorted list of Model instances.
        """
        target = library.path
        if not target.is_dir():
            return []
        files = sorted(
            p
            for p in target.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in VALID_MODEL_EXTENSIONS
        )
        return [
            Model(
                name=f.stem,
                path=Path(f.name),
            )
            for f in files
        ]

    def get_all_models(self) -> List[Model]:
        """
        List all models across all libraries.

        Deduplicates by filename — earlier libraries take precedence.

        Returns:
            Sorted list of Model instances.
        """
        seen_filenames: set = set()
        combined: List[Model] = []
        for lib in self._libraries.values():
            models = self.get_models(lib)
            for m in models:
                if m.path.name not in seen_filenames:
                    combined.append(m)
                    seen_filenames.add(m.path.name)
        return sorted(combined, key=lambda m: m.name)

    def _get_bundled_path(self) -> Optional[Path]:
        try:
            from rayforge.resources import models as resource_models

            p = Path(str(importlib.resources.files(resource_models)))
            if p.is_dir():
                return p
        except (FileNotFoundError, ModuleNotFoundError):
            pass
        return None

    def __str__(self) -> str:
        return f"ModelManager(libraries={len(self._libraries)})"
