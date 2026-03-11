"""Library manager for material libraries in Rayforge."""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from blinker import Signal
from .material import Material
from .material_library import MaterialLibrary

logger = logging.getLogger(__name__)


class LibraryManager:
    """
    Application-wide manager for material libraries.

    Manages multiple MaterialLibrary instances. User libraries are
    stored in the user_dir and are writable. Addon libraries are
    registered via add_library() and are typically read-only.
    """

    def __init__(self, user_dir: Path):
        """
        Initialize the library manager.

        Args:
            user_dir: Directory for user materials (writable)
        """
        self.user_dir = user_dir
        self.libraries_changed = Signal()

        self._libraries: Dict[str, MaterialLibrary] = {}
        # Track which addon registered each library (library_id -> addon_name)
        self._library_addons: Dict[str, str] = {}

        self.user_dir.mkdir(parents=True, exist_ok=True)

    def load_all_libraries(self) -> None:
        """Load all user material libraries from user_dir."""
        self._libraries.clear()

        for path in self.user_dir.iterdir():
            if path.is_dir():
                user_library = MaterialLibrary(path, read_only=False)
                user_library.load_materials()
                lib_id = user_library.library_id
                self._libraries[lib_id] = user_library
                logger.debug(f"Loaded user library: {lib_id}")

        logger.info(f"Loaded {len(self._libraries)} material libraries")

    def create_user_library(self, display_name: str) -> Optional[str]:
        """
        Creates a new user library and returns its ID.

        Args:
            display_name: The human-readable name for the new library.

        Returns:
            The unique ID of the new library, or None on failure.
        """
        if not display_name:
            return None

        # Generate a unique directory name
        temp_id = str(uuid.uuid4())
        lib_dir = self.user_dir / temp_id

        # Delegate library creation to MaterialLibrary
        library = MaterialLibrary.create(lib_dir, display_name)

        if library is not None:
            # Use the library's own ID as the key
            lib_id = library.library_id
            self._libraries[lib_id] = library
            self.libraries_changed.send(self)
            return lib_id
        else:
            return None

    def remove_user_library(self, library_id: str) -> bool:
        """Removes a writable library by its ID."""
        if library_id not in self._libraries:
            logger.error(f"Library with ID '{library_id}' not found.")
            return False

        library = self._libraries[library_id]
        if library.read_only:
            logger.error(f"Cannot remove read-only library '{library_id}'.")
            return False

        try:
            shutil.rmtree(library._directory)
            del self._libraries[library_id]
            logger.info(f"Removed library: {library.display_name}")
            self.libraries_changed.send(self)
            return True
        except OSError as e:
            logger.error(f"Failed to remove library '{library_id}': {e}")
            return False

    def update_library(self, library_id: str) -> bool:
        """
        Save library changes to disk by delegating to the library's save
        method.

        Args:
            library_id: ID of the library to update

        Returns:
            True if updated successfully, False otherwise
        """
        if library_id not in self._libraries:
            logger.error(f"Library with ID '{library_id}' not found.")
            return False

        library = self._libraries[library_id]
        return library.save()

    def get_library(self, library_id: str) -> Optional[MaterialLibrary]:
        """
        Get a library by ID.

        Args:
            library_id: ID of the library

        Returns:
            MaterialLibrary instance or None if not found
        """
        if not self._libraries:
            self.load_all_libraries()

        return self._libraries.get(library_id)

    def get_libraries(self) -> List[MaterialLibrary]:
        """
        Get all libraries.

        Returns:
            List of all MaterialLibrary instances
        """
        if not self._libraries:
            self.load_all_libraries()

        return list(self._libraries.values())

    def get_material(self, uid: str) -> Optional[Material]:
        """
        Get a material by UID, searching all libraries.

        Args:
            uid: Unique identifier of the material

        Returns:
            Material instance or None if not found
        """
        if not self._libraries:
            self.load_all_libraries()

        # Prioritize writable libraries over read-only ones.
        sorted_libs = sorted(
            self._libraries.values(),
            key=lambda lib: (
                lib.read_only,
                lib.library_id,
            ),
        )

        for library in sorted_libs:
            material = library.get_material(uid)
            if material:
                return material

        return None

    def get_material_or_none(self, uid: str) -> Optional[Material]:
        """
        Get a material by UID with graceful fallback.

        This method never raises an exception and always returns
        either a Material instance or None.

        Args:
            uid: Unique identifier of the material

        Returns:
            Material instance or None if not found
        """
        try:
            return self.get_material(uid)
        except Exception as e:
            logger.warning(f"Error getting material {uid}: {e}")
            return None

    def resolve_material(self, uid: str) -> Optional[Material]:
        """
        Resolve a material reference with fallback handling.

        Similar to get_material_or_none but with additional logging
        for debugging missing material references.

        Args:
            uid: Unique identifier of the material

        Returns:
            Material instance or None if not found
        """
        material = self.get_material_or_none(uid)

        if material is None and uid:
            logger.debug(f"Material reference '{uid}' could not be resolved")

        return material

    def add_material(self, material: Material, library_id: str) -> bool:
        """
        Add a material to a library.

        Args:
            material: Material to add
            library_id: ID of the library to add to.

        Returns:
            True if added successfully, False otherwise
        """
        if not self._libraries:
            self.load_all_libraries()

        if library_id not in self._libraries:
            logger.error(f"Library '{library_id}' not found")
            return False

        library = self._libraries[library_id]
        return library.add_material(material)

    def remove_material(self, uid: str, library_id: str) -> bool:
        """
        Remove a material from a library.

        Args:
            uid: Unique identifier of the material
            library_id: ID of the library to remove from.

        Returns:
            True if removed successfully, False otherwise
        """
        if not self._libraries:
            self.load_all_libraries()

        if library_id not in self._libraries:
            logger.error(f"Library '{library_id}' not found")
            return False

        library = self._libraries[library_id]
        return library.remove_material(uid)

    def get_all_materials(self) -> List[Material]:
        """
        Get all materials from all libraries.

        Returns:
            List of all materials, with user materials first
        """
        if not self._libraries:
            self.load_all_libraries()
        all_materials = []

        # Add writable library materials first
        for lib in self.get_libraries():
            if not lib.read_only:
                all_materials.extend(lib.get_all_materials())

        # Add materials from read-only libraries
        for lib in self.get_libraries():
            if lib.read_only:
                all_materials.extend(lib.get_all_materials())

        return all_materials

    def reload_libraries(self) -> None:
        """Reload all libraries from disk."""
        self.load_all_libraries()
        self.libraries_changed.send(self)
        logger.info("Reloaded all material libraries")

    def get_library_ids(self) -> List[str]:
        """
        Get the IDs of all libraries.

        Returns:
            List of library IDs
        """
        if not self._libraries:
            self.load_all_libraries()

        return list(self._libraries.keys())

    def add_library(
        self,
        library: MaterialLibrary,
        addon_name: Optional[str] = None,
        allow_overwrite: bool = False,
    ) -> bool:
        """
        Add a MaterialLibrary to the manager.

        Args:
            library: The MaterialLibrary instance to add
            addon_name: Optional addon name for tracking ownership
            allow_overwrite: If True, allows replacing an existing library
                with the same ID

        Returns:
            True if added successfully, False otherwise
        """
        lib_id = library.library_id
        if lib_id in self._libraries and not allow_overwrite:
            logger.error(f"Library with ID '{lib_id}' already exists")
            return False

        self._libraries[lib_id] = library
        if addon_name:
            self._library_addons[lib_id] = addon_name
        logger.info(f"Added library: {library.display_name} ({lib_id})")
        self.libraries_changed.send(self)
        return True

    def add_library_from_path(
        self,
        path: Path,
        read_only: bool = True,
        addon_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create and add a MaterialLibrary from a directory path.

        This is a convenience method for addons to register material
        libraries from a filesystem path.

        Args:
            path: Directory containing material YAML files
            read_only: If True (default), the library cannot be modified
            addon_name: Optional addon name for tracking ownership

        Returns:
            The library ID if added successfully, None otherwise
        """
        if not path.exists() or not path.is_dir():
            logger.error(
                f"Library path does not exist or is not a directory: {path}"
            )
            return None

        library = MaterialLibrary(path, read_only=read_only)
        library.load_materials()

        if self.add_library(library, addon_name=addon_name):
            return library.library_id
        return None

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Remove all libraries registered by a specific addon.

        Args:
            addon_name: The name of the addon whose libraries should be removed

        Returns:
            The number of libraries removed
        """
        lib_ids_to_remove = [
            lib_id
            for lib_id, name in self._library_addons.items()
            if name == addon_name
        ]

        for lib_id in lib_ids_to_remove:
            if lib_id in self._libraries:
                del self._libraries[lib_id]
            del self._library_addons[lib_id]

        if lib_ids_to_remove:
            logger.info(
                f"Removed {len(lib_ids_to_remove)} libraries from addon "
                f"'{addon_name}'"
            )
            self.libraries_changed.send(self)

        return len(lib_ids_to_remove)

    def __len__(self) -> int:
        """Get the total number of materials across all libraries."""
        return len(self.get_all_materials())

    def __str__(self) -> str:
        """String representation of the library manager."""
        if not self._libraries:
            self.load_all_libraries()
        return (
            f"LibraryManager(libraries={len(self._libraries)}, "
            f"materials={len(self)})"
        )
