"""Material library management for Rayforge."""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from .material import Material

logger = logging.getLogger(__name__)


class MaterialLibrary:
    """
    Represents a single material library from a specific source.

    Manages loading materials from a directory and handles read-only
    libraries (core materials) vs user libraries.
    """

    def __init__(self, directory: Path, source: str):
        """
        Initialize a material library.

        Args:
            directory: Directory containing material files
            source: The source of the library ('core', 'user', plugin name)
        """
        self._directory = directory
        self.source = source
        self._materials: Dict[str, Material] = {}
        self._loaded = False
        self._display_name: str = ""
        self._library_id: str = ""

    @property
    def display_name(self) -> str:
        """Get the human-readable display name for the library."""
        if not self._loaded:
            self.load_materials()

        return self._display_name or self._directory.name

    def set_display_name(self, display_name: str) -> None:
        """
        Set the display name for the library.

        This method only updates the in-memory display name. To persist
        the change, the library must be saved by calling save().

        Args:
            display_name: New display name for the library
        """
        self._display_name = display_name

    @property
    def library_id(self) -> str:
        """Get the library ID from metadata."""
        if not self._loaded:
            self.load_materials()
        return self._library_id or self._directory.name

    @property
    def read_only(self) -> bool:
        """A library is read-only if its source is not 'user'."""
        return self.source != "user"

    @property
    def is_loaded(self) -> bool:
        """Check if the library has been loaded."""
        return self._loaded

    @classmethod
    def create(
        cls, directory: Path, display_name: str
    ) -> Optional["MaterialLibrary"]:
        """
        Create a new material library with the given display name.

        This class method handles the creation of the directory structure
        and metadata file for a new library.

        Args:
            directory: Directory where the library should be created
            display_name: The human-readable name for the new library

        Returns:
            MaterialLibrary instance if created successfully, None otherwise
        """
        logger.debug(
            f"MaterialLibrary.create called with directory={directory}, "
            f"display_name={display_name}"
        )

        if not display_name:
            logger.error("Cannot create library with empty display name")
            return None

        if directory.exists():
            logger.error(f"Library directory '{directory}' already exists")
            return None

        try:
            # Create the directory
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

            # Create library with a generated UUID
            import uuid

            lib_id = str(uuid.uuid4())
            logger.debug(f"Generated library ID: {lib_id}")

            library = cls(directory, source="user")
            library._display_name = display_name
            library._library_id = lib_id
            library._loaded = (
                True  # Mark as loaded since we're setting the values directly
            )

            # Save the metadata
            save_result = library.save()
            logger.debug(f"Library.save() returned: {save_result}")

            if save_result:
                logger.info(
                    f"Created new user library: {display_name} ({lib_id})"
                )
                return library
            else:
                # Clean up directory if save failed
                import shutil

                shutil.rmtree(directory)
                logger.error("Library save failed, cleaned up directory")
                return None

        except OSError as e:
            logger.error(f"Failed to create library '{display_name}': {e}")
            return None

    def load_materials(self) -> None:
        """
        Load all materials from the library directory.

        Scans the directory for .yaml files and loads them as materials.
        Invalid files are skipped with warnings.
        """
        if self._loaded:
            return

        self._materials.clear()
        self._display_name = ""

        if not self._directory.exists():
            if not self.read_only:
                self._directory.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"Created material library directory: {self._directory}"
                )
            else:
                logger.warning(
                    f"Material library directory not found: {self._directory}"
                )
                return

        # Load library metadata first
        meta_file = self._directory / "__library__.yaml"
        if meta_file.is_file():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta_data = yaml.safe_load(f)
                if isinstance(meta_data, dict):
                    self._display_name = meta_data.get("name", "")
                    self._library_id = meta_data.get("id", "")
            except Exception as e:
                logger.warning(
                    f"Could not load library metadata from {meta_file}: {e}"
                )

        # Load all YAML files in the directory
        for file_path in self._directory.glob("*.yaml"):
            if file_path.name == "__library__.yaml":
                continue  # Skip metadata file

            try:
                material = Material.from_file(file_path)
                self._materials[material.uid] = material
                logger.debug(f"Loaded material: {material.uid}")
            except Exception as e:
                logger.warning(
                    f"Failed to load material from {file_path}: {e}"
                )

        self._loaded = True
        logger.info(
            f"Loaded {len(self._materials)} materials from "
            f"{self._directory.name}"
        )

    def get_material(self, uid: str) -> Optional[Material]:
        """
        Get a material by UID.

        Args:
            uid: Unique identifier of the material

        Returns:
            Material instance or None if not found
        """
        if not self._loaded:
            self.load_materials()

        return self._materials.get(uid)

    def get_all_materials(self) -> List[Material]:
        """
        Get all materials in the library.

        Returns:
            List of all materials
        """
        if not self._loaded:
            self.load_materials()

        return list(self._materials.values())

    def add_material(self, material: Material) -> bool:
        """
        Add a material to the library.

        Args:
            material: Material to add

        Returns:
            True if added successfully, False if read-only or already exists
        """
        if self.read_only:
            logger.warning(
                f"Cannot add material to read-only library: "
                f"{self._directory.name}"
            )
            return False

        if material.uid in self._materials:
            logger.warning(
                f"Material {material.uid} already exists in "
                f"{self._directory.name}"
            )
            return False

        # Save material to file
        file_path = self._directory / f"{material.uid}.yaml"
        try:
            material.save_to_file(file_path)
            self._materials[material.uid] = material
            logger.info(
                f"Added material {material.uid} to {self._directory.name}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save material {material.uid}: {e}")
            return False

    def remove_material(self, uid: str) -> bool:
        """
        Remove a material from the library.

        Args:
            uid: Unique identifier of the material to remove

        Returns:
            True if removed successfully, False if read-only or not found
        """
        if self.read_only:
            logger.warning(
                f"Cannot remove material from read-only library: "
                f"{self._directory.name}"
            )
            return False

        if uid not in self._materials:
            logger.warning(
                f"Material {uid} not found in {self._directory.name}"
            )
            return False

        material = self._materials[uid]

        # Remove file if it exists
        if material.file_path and material.file_path.exists():
            try:
                material.file_path.unlink()
                logger.info(f"Removed material file: {material.file_path}")
            except Exception as e:
                logger.error(f"Failed to remove material file: {e}")
                return False

        # Remove from memory
        del self._materials[uid]
        logger.info(f"Removed material {uid} from {self._directory.name}")
        return True

    def save(self) -> bool:
        """
        Save library metadata to disk.

        This method persists the current state of the library to its
        metadata file. The library directory itself is never renamed.

        Returns:
            True if saved successfully, False otherwise
        """
        if self.source == "core":
            logger.warning(f"Cannot save core library: {self._directory.name}")
            return False

        try:
            meta_file = self._directory / "__library__.yaml"

            # Read existing metadata if it exists
            existing_data = {}
            if meta_file.is_file():
                with open(meta_file, "r", encoding="utf-8") as f:
                    existing_data = yaml.safe_load(f) or {}

            # Update with current library data
            existing_data["name"] = self._display_name or self._directory.name
            existing_data["id"] = self.library_id

            # Write updated metadata
            with open(meta_file, "w", encoding="utf-8") as f:
                yaml.dump(existing_data, f, sort_keys=False)

            logger.info(f"Saved library: {self.display_name}")
            return True
        except OSError as e:
            logger.error(f"Failed to save library: {e}")
            return False

    def reload(self) -> None:
        """Reload all materials from the directory."""
        self._loaded = False
        self.load_materials()

    def __len__(self) -> int:
        """Get the number of materials in the library."""
        if not self._loaded:
            self.load_materials()
        return len(self._materials)

    def __contains__(self, uid: str) -> bool:
        """Check if a material UID exists in the library."""
        if not self._loaded:
            self.load_materials()
        return uid in self._materials

    def __iter__(self):
        """Iterate over materials in the library."""
        if not self._loaded:
            self.load_materials()
        return iter(self._materials.values())

    def __str__(self) -> str:
        """String representation of the library."""
        return (
            f"MaterialLibrary(name='{self._directory.name}', "
            f"materials={len(self)}, "
            f"read_only={self.read_only})"
        )
