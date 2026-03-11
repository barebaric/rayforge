"""Tests for the LibraryManager class."""

import tempfile
import yaml
from pathlib import Path
from rayforge.core.material import Material
from rayforge.core.library_manager import LibraryManager


class TestLibraryManager:
    """Test cases for the LibraryManager class."""

    def test_manager_creation(self):
        """Test creating a LibraryManager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            manager = LibraryManager(user_dir)

            assert manager.user_dir == user_dir
            assert len(manager) == 0

    def test_manager_load_libraries(self):
        """Test loading libraries in the manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            # Create user subdirectory with materials
            user_dir.mkdir(parents=True, exist_ok=True)
            user_subdir = user_dir / "test_lib"
            user_subdir.mkdir()
            material2_data = {
                "uid": "material2",
                "name": "Material 2",
                "category": "test",
            }
            with open(user_subdir / "material2.yaml", "w") as f:
                yaml.dump(material2_data, f)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Should have one library (user subdirectory)
            libraries = manager.get_libraries()
            assert len(libraries) == 1

            # Should have one material total
            assert len(manager) == 1

            # Should be able to get materials
            assert manager.get_material("material2") is not None

    def test_manager_get_library(self):
        """Test getting a library by ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            # Create a library with metadata
            user_dir.mkdir(parents=True, exist_ok=True)
            user_subdir = user_dir / "test_lib"
            user_subdir.mkdir()
            metadata = {"name": "Test Library", "id": "test-lib-id"}
            with open(user_subdir / "__library__.yaml", "w") as f:
                yaml.dump(metadata, f)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Should have the library
            library = manager.get_library("test-lib-id")
            assert library is not None
            assert library.read_only is False

            # Non-existent library should return None
            assert manager.get_library("nonexistent") is None

    def test_manager_get_material_or_none(self):
        """Test getting a material with graceful fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            # Create a material in a user subdirectory
            user_dir.mkdir(parents=True, exist_ok=True)
            user_subdir = user_dir / "test_lib"
            user_subdir.mkdir()
            material_data = {
                "uid": "material1",
                "name": "Material 1",
                "category": "test",
            }
            with open(user_subdir / "material1.yaml", "w") as f:
                yaml.dump(material_data, f)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Should get the material
            material = manager.get_material_or_none("material1")
            assert material is not None
            assert material.uid == "material1"

            # Non-existent material should return None
            assert manager.get_material_or_none("nonexistent") is None

    def test_manager_resolve_material(self):
        """Test resolving a material reference with fallback handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Non-existent material should return None
            material = manager.resolve_material("nonexistent")
            assert material is None

    def test_manager_add_material(self):
        """Test adding a material to a library."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Create a user library first
            lib_id = manager.create_user_library("Test Library")
            assert lib_id is not None
            manager.reload_libraries()

            # Create a material
            material = Material(
                uid="material1", name="Material 1", category="test"
            )

            # Add to user library
            result = manager.add_material(material, lib_id)

            assert result is True
            assert len(manager) == 1
            assert manager.get_material("material1") is not None

            # Try to add to non-existent library
            result = manager.add_material(material, "nonexistent")
            assert result is False

    def test_manager_remove_material(self):
        """Test removing a material from a library."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Create a user library first
            lib_id = manager.create_user_library("Test Library")
            assert lib_id is not None
            manager.reload_libraries()

            # Create a material
            material = Material(
                uid="material1", name="Material 1", category="test"
            )

            # Add to user library
            manager.add_material(material, lib_id)
            assert len(manager) == 1

            # Remove from user library
            result = manager.remove_material("material1", lib_id)

            assert result is True
            assert len(manager) == 0
            assert manager.get_material("material1") is None

            # Try to remove from non-existent library
            result = manager.remove_material("material1", "nonexistent")
            assert result is False

    def test_manager_get_all_materials(self):
        """Test getting all materials from all libraries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            # Create user subdirectory with materials
            user_dir.mkdir(parents=True, exist_ok=True)
            user_subdir = user_dir / "test_lib"
            user_subdir.mkdir()
            material2_data = {
                "uid": "material2",
                "name": "Material 2",
                "category": "test",
            }
            with open(user_subdir / "material2.yaml", "w") as f:
                yaml.dump(material2_data, f)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Get all materials
            all_materials = manager.get_all_materials()
            assert len(all_materials) == 1
            assert all_materials[0].uid == "material2"

    def test_manager_reload_libraries(self):
        """Test reloading all libraries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Initially empty
            assert len(manager) == 0

            # Add a material file directly to a user subdirectory
            user_subdir = user_dir / "test_lib"
            user_subdir.mkdir()
            material_data = {
                "uid": "material1",
                "name": "Material 1",
                "category": "test",
            }
            with open(user_subdir / "material1.yaml", "w") as f:
                yaml.dump(material_data, f)

            # Reload all libraries
            manager.reload_libraries()

            # Should now have the material
            assert len(manager) == 1
            assert manager.get_material("material1") is not None

    def test_manager_get_library_names(self):
        """Test getting the IDs of all libraries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            # Create a library with metadata
            user_dir.mkdir(parents=True, exist_ok=True)
            user_subdir = user_dir / "test_lib"
            user_subdir.mkdir()
            metadata = {"name": "Test Library", "id": "test-lib-id"}
            with open(user_subdir / "__library__.yaml", "w") as f:
                yaml.dump(metadata, f)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Get library IDs
            library_ids = manager.get_library_ids()
            assert "test-lib-id" in library_ids
            assert len(library_ids) == 1

    def test_manager_str_repr(self):
        """Test string representations of LibraryManager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            assert "LibraryManager" in str(manager)
            assert "libraries=0" in str(manager)
            assert "materials=0" in str(manager)

    def test_manager_create_user_library(self):
        """Test creating a new user library."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Initially should have no libraries
            assert len(manager.get_library_ids()) == 0

            # Create a new user library
            lib_id = manager.create_user_library("Test Library")
            assert lib_id is not None

            # Reload to pick up the new library
            manager.reload_libraries()

            # Should now have 1 library
            library_ids = manager.get_library_ids()
            assert len(library_ids) == 1
            assert lib_id in library_ids

            # Get the new library and check its properties
            new_library = manager.get_library(lib_id)
            assert new_library is not None
            assert new_library.read_only is False
            assert new_library.display_name == "Test Library"

            # Try to create a library with empty name
            empty_id = manager.create_user_library("")
            assert empty_id is None

    def test_manager_remove_user_library(self):
        """Test removing a writable library."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Create a new user library
            lib_id = manager.create_user_library("Test Library")
            assert lib_id is not None
            manager.reload_libraries()

            # Add a material to the new library
            material = Material(
                uid="test_material", name="Test Material", category="test"
            )
            manager.add_material(material, lib_id)
            assert len(manager) == 1

            # Remove the library
            result = manager.remove_user_library(lib_id)
            assert result is True

            # Reload to reflect the removal
            manager.reload_libraries()

            # Library should be gone
            assert manager.get_library(lib_id) is None
            assert lib_id not in manager.get_library_ids()

            # Material should be gone too
            assert len(manager) == 0

            # Try to remove non-existent library
            result = manager.remove_user_library("nonexistent")
            assert result is False

    def test_manager_update_library(self):
        """Test updating a library's display name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Create a new user library
            lib_id = manager.create_user_library("Test Library")
            assert lib_id is not None
            manager.reload_libraries()

            # Get the library and change its display name
            library = manager.get_library(lib_id)
            assert library is not None
            assert library.display_name == "Test Library"

            # Update the display name directly
            library.set_display_name("Updated Library Name")

            # Save the changes
            result = manager.update_library(lib_id)
            assert result is True

            # Reload and verify the change persisted
            manager.reload_libraries()
            updated_library = manager.get_library(lib_id)
            assert updated_library is not None
            assert updated_library.display_name == "Updated Library Name"

            # Try to update non-existent library
            result = manager.update_library("nonexistent")
            assert result is False

    def test_manager_add_library(self):
        """Test adding a MaterialLibrary directly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir) / "user"
            user_dir.mkdir()
            library_dir = Path(temp_dir) / "addon_lib"
            library_dir.mkdir()

            # Create a material file
            material_data = {
                "uid": "material1",
                "name": "Material 1",
                "category": "test",
            }
            with open(library_dir / "material1.yaml", "w") as f:
                yaml.dump(material_data, f)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Create and add a library
            from rayforge.core.material_library import MaterialLibrary

            library = MaterialLibrary(library_dir, read_only=True)
            library.load_materials()

            result = manager.add_library(library)
            assert result is True
            assert manager.get_library(library.library_id) is not None

            # Try to add duplicate (should fail)
            result = manager.add_library(library)
            assert result is False

    def test_manager_add_library_with_addon_name(self):
        """Test adding a library with addon name tracking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir) / "user"
            user_dir.mkdir()
            library_dir = Path(temp_dir) / "addon_lib"
            library_dir.mkdir()

            manager = LibraryManager(user_dir)

            from rayforge.core.material_library import MaterialLibrary

            library = MaterialLibrary(library_dir, read_only=True)
            library.load_materials()

            result = manager.add_library(library, addon_name="test_addon")
            assert result is True

            # Remove libraries by addon name
            count = manager.unregister_all_from_addon("test_addon")
            assert count == 1
            assert manager.get_library(library.library_id) is None

    def test_manager_add_library_from_path(self):
        """Test adding a library from a path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)
            library_dir = Path(temp_dir) / "addon_lib"
            library_dir.mkdir()

            # Create a material file
            material_data = {
                "uid": "material1",
                "name": "Material 1",
                "category": "test",
            }
            with open(library_dir / "material1.yaml", "w") as f:
                yaml.dump(material_data, f)

            manager = LibraryManager(user_dir)

            lib_id = manager.add_library_from_path(library_dir, read_only=True)
            assert lib_id is not None

            library = manager.get_library(lib_id)
            assert library is not None
            assert library.read_only is True
            assert len(library) == 1

            # Try to add from non-existent path
            lib_id = manager.add_library_from_path(
                Path(temp_dir) / "nonexistent"
            )
            assert lib_id is None

    def test_manager_material_priority(self):
        """Test that writable library materials take priority."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir) / "user"
            addon_dir = Path(temp_dir) / "addon"

            # Create writable library with material
            user_dir.mkdir(parents=True, exist_ok=True)
            user_subdir = user_dir / "test_lib"
            user_subdir.mkdir()
            material_data = {
                "uid": "duplicate_material",
                "name": "Writable Material",
                "category": "test",
            }
            with open(user_subdir / "duplicate_material.yaml", "w") as f:
                yaml.dump(material_data, f)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Create a read-only library with same material UID
            addon_lib_dir = addon_dir / "addon_lib"
            addon_lib_dir.mkdir(parents=True, exist_ok=True)
            material_data2 = {
                "uid": "duplicate_material",
                "name": "Read-Only Material",
                "category": "test",
            }
            with open(addon_lib_dir / "duplicate_material.yaml", "w") as f:
                yaml.dump(material_data2, f)

            manager.add_library_from_path(addon_lib_dir, read_only=True)

            # Should get the writable library material (priority)
            material = manager.get_material("duplicate_material")
            assert material is not None
            assert material.name == "Writable Material"

    def test_manager_user_library_with_id(self):
        """Test that user libraries are created with ID in metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            user_dir = Path(temp_dir)

            manager = LibraryManager(user_dir)
            manager.load_all_libraries()

            # Create a new user library
            lib_id = manager.create_user_library("Test Library")
            assert lib_id is not None

            # Reload to pick up the new library
            manager.reload_libraries()

            # Get the new library and check its properties
            new_library = manager.get_library(lib_id)
            assert new_library is not None
            assert new_library.library_id == lib_id

            # Check that __library__.yaml was created with the ID
            # Access the private directory for testing purposes only
            meta_file = new_library._directory / "__library__.yaml"
            assert meta_file.exists()
            with open(meta_file, "r") as f:
                metadata = yaml.safe_load(f)
            assert metadata["id"] == lib_id
            assert metadata["name"] == "Test Library"
