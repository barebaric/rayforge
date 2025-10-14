"""Library manager for material libraries in Rayforge."""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from .material import Material
from .material_library import MaterialLibrary

logger = logging.getLogger(__name__)


class LibraryManager:
    """
    Application-wide manager for material libraries.

    Follows the same pattern as MachineManager and ConfigManager,
    managing multiple MaterialLibrary instances and handling both
    user and core material directories.
    """

    def __init__(self, core_dir: Path, user_dir: Path):
        """
        Initialize the library manager.

        Args:
            core_dir: Directory for core (read-only) materials
            user_dir: Directory for user materials
        """
        self.core_dir = core_dir
        self.user_dir = user_dir

        # Create libraries
        self._libraries: Dict[str, MaterialLibrary] = {}

        # Ensure directories exist
        self.core_dir.mkdir(parents=True, exist_ok=True)
        self.user_dir.mkdir(parents=True, exist_ok=True)

    def load_all_libraries(self) -> None:
        """Load all material libraries from core and user directories."""
        self._libraries.clear()

        # Load core library (treat core_dir as a single library)
        core_library = MaterialLibrary(self.core_dir, source="core")
        core_library.load_materials()
        core_id = core_library.library_id or "core"
        self._libraries[core_id] = core_library
        logger.debug(f"Loaded core library: {core_id}")

        # Load user libraries from subdirectories of user_dir
        for path in self.user_dir.iterdir():
            if path.is_dir():
                user_sub_library = MaterialLibrary(path, source="user")
                user_sub_library.load_materials()
                lib_id = user_sub_library.library_id
                self._libraries[lib_id] = user_sub_library
                logger.debug(f"Loaded user sub-library: {lib_id}")

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
            return lib_id
        else:
            return None

    def remove_user_library(self, library_id: str) -> bool:
        """Removes a user library by its ID."""
        if library_id not in self._libraries:
            logger.error(f"Library with ID '{library_id}' not found.")
            return False

        library = self._libraries[library_id]
        if library.source != "user":
            logger.error(f"Cannot remove non-user library '{library_id}'.")
            return False

        try:
            shutil.rmtree(library._directory)
            del self._libraries[library_id]
            logger.info(f"Removed user library: {library.display_name}")
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

        # Prioritize user libraries, then core, then others.
        sorted_libs = sorted(
            self._libraries.values(),
            key=lambda lib: (
                lib.source != "user",
                lib.source != "core",
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

        # Add user materials first
        for lib in self.get_libraries():
            if lib.source == "user":
                all_materials.extend(lib.get_all_materials())

        # Add materials from other libraries
        for lib in self.get_libraries():
            if lib.source != "user":
                all_materials.extend(lib.get_all_materials())

        return all_materials

    def reload_libraries(self) -> None:
        """Reload all libraries from disk."""
        self.load_all_libraries()
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
