"""Model manager for 3D model assets in Rayforge."""

import importlib.resources
import logging
import shutil
from gettext import gettext as _
from pathlib import Path
from typing import Dict, List, Optional

from blinker import Signal

from ..addon_mgr.addon_manager import AddonRegistry
from .model import Model, ModelCategory, ModelLibrary

logger = logging.getLogger(__name__)

VALID_MODEL_EXTENSIONS = {".glb", ".gltf"}

CATEGORIES: List[ModelCategory] = [
    ModelCategory(
        id="machines",
        display_name=_("Machines"),
        description=_("Complete machine models."),
    ),
    ModelCategory(
        id="rotary",
        display_name=_("Rotary"),
        description=_("Rotary and workholding models."),
    ),
    ModelCategory(
        id="heads",
        display_name=_("Laser Heads"),
        description=_("Laser head models."),
    ),
]


class ModelManager(AddonRegistry):
    """
    Application-wide manager for 3D model assets.

    Maintains an ordered collection of :class:`ModelLibrary`
    instances.  When resolving a model path the libraries are
    searched in registration order, so earlier libraries take
    precedence.

    The ``user`` library is always registered first and points to the
    user-writable directory.  Additional libraries (bundled resources,
    addon-contributed) are registered via :meth:`add_library` or
    :meth:`register_bundled_library`.
    """

    def __init__(self, user_dir: Path):
        self.user_dir = user_dir
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.changed = Signal()
        self._libraries: Dict[str, ModelLibrary] = {}
        self._library_addons: Dict[str, Optional[str]] = {}
        self.add_library(
            ModelLibrary(
                library_id="user",
                display_name=_("User"),
                path=self.user_dir,
                read_only=False,
                icon_name="home-symbolic",
            )
        )

    @staticmethod
    def get_categories() -> List[ModelCategory]:
        """Return the fixed list of model categories."""
        return list(CATEGORIES)

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
        returned directly if the file exists.  Falls back to
        ``importlib.resources`` for backward compatibility with
        paths that carry a ``models/`` prefix.
        """
        path = model.path

        if path.is_absolute():
            return path if path.is_file() else None

        for lib in self._libraries.values():
            full = lib.path / path
            if full.is_file():
                return full

        return self._resolve_bundled(path)

    def get_user_models(self, subdir: Path = Path("")) -> List[Model]:
        """
        List user models in the given subdirectory.

        Args:
            subdir: Subdirectory under the user models dir to list.

        Returns:
            List of Model instances for each file found.
        """
        user_lib = self._libraries.get("user")
        if user_lib is None:
            return []
        target = user_lib.path / subdir
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
                path=f.relative_to(user_lib.path),
                file_path=f,
            )
            for f in files
        ]

    def get_models(
        self, library: ModelLibrary, category: ModelCategory
    ) -> List[Model]:
        """
        List models for *category* inside *library*.

        Args:
            library: The ModelLibrary to query.
            category: The ModelCategory to list models for.

        Returns:
            Sorted list of Model instances.
        """
        subdir = Path(category.id)
        target = library.path / subdir
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
                path=subdir / f.name,
                file_path=f,
            )
            for f in files
        ]

    def get_all_models(self, category: ModelCategory) -> List[Model]:
        """
        List all models for a category across all libraries.

        Deduplicates by filename -- earlier libraries take precedence.

        Args:
            category: The ModelCategory to list models for.

        Returns:
            Sorted list of Model instances.
        """
        seen_filenames: set = set()
        combined: List[Model] = []
        for lib in self._libraries.values():
            models = self.get_models(lib, category)
            for m in models:
                if m.path.name not in seen_filenames:
                    combined.append(m)
                    seen_filenames.add(m.path.name)
        return sorted(combined, key=lambda m: m.name)

    def is_user_model(self, model: Model) -> bool:
        """Return True if the model lives in the user dir."""
        if model.file_path is None:
            return False
        try:
            model.file_path.relative_to(self.user_dir)
            return True
        except ValueError:
            return False

    def add_model(self, src: Path, model: Model) -> Model:
        """
        Copy a model file into the user models directory.

        Args:
            src: Source file path to copy.
            model: Model instance whose ``path`` field is used as
                the relative destination under user_dir.

        Returns:
            A new Model instance with ``file_path`` set.
        """
        dest = self.user_dir / model.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        self.changed.send(self)
        return Model(
            name=model.name,
            path=model.path,
            description=model.description,
            file_path=dest,
            extra=model.extra,
        )

    def remove_model(self, model: Model) -> bool:
        """
        Remove a model file from the user models directory.

        Returns:
            True if removed, False otherwise.
        """
        target = self.user_dir / model.path
        if not target.is_file():
            return False
        target.unlink()
        self.changed.send(self)
        return True

    def _get_bundled_path(self) -> Optional[Path]:
        try:
            from rayforge.resources import models as resource_models

            p = Path(str(importlib.resources.files(resource_models)))
            if p.is_dir():
                return p
        except (FileNotFoundError, ModuleNotFoundError):
            pass
        return None

    def _resolve_bundled(self, relative_path: Path) -> Optional[Path]:
        parts = relative_path.parts
        if parts and parts[0] == "models":
            resource_subpath = Path(*parts[1:])
        else:
            resource_subpath = relative_path

        try:
            from rayforge.resources import models as resource_models

            resource_file = importlib.resources.files(
                resource_models
            ).joinpath(*resource_subpath.parts)
            if resource_file.is_file():
                return Path(str(resource_file))
        except (FileNotFoundError, TypeError, ModuleNotFoundError):
            pass

        return None

    def __str__(self) -> str:
        return f"ModelManager(user_dir={self.user_dir})"
