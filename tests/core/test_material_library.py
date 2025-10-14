"""Tests for the MaterialLibrary class."""

import tempfile
import yaml
from pathlib import Path
from rayforge.core.material import Material
from rayforge.core.material_library import MaterialLibrary


class TestMaterialLibrary:
    """Test cases for the MaterialLibrary class."""

    def test_library_creation(self):
        """Test creating a MaterialLibrary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)
            library = MaterialLibrary(library_dir, source="user")

            assert library.read_only is False
            assert library.is_loaded is False
            assert library.source == "user"

    def test_library_load_empty_directory(self):
        """Test loading a library from an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)
            library = MaterialLibrary(library_dir, source="user")

            library.load_materials()

            assert library.is_loaded is True
            assert len(library) == 0
            assert library.get_all_materials() == []

    def test_library_load_with_materials(self):
        """Test loading a library with material files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)

            # Create material files
            material1_data = {
                "uid": "material1",
                "name": "Material 1",
                "category": "test",
            }
            material2_data = {
                "uid": "material2",
                "name": "Material 2",
                "category": "test",
            }

            with open(library_dir / "material1.yaml", "w") as f:
                yaml.dump(material1_data, f)
            with open(library_dir / "material2.yaml", "w") as f:
                yaml.dump(material2_data, f)

            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            assert library.is_loaded is True
            assert len(library) == 2

            materials = library.get_all_materials()
            uids = [m.uid for m in materials]
            assert "material1" in uids
            assert "material2" in uids

    def test_library_load_invalid_file(self):
        """Test loading a library with an invalid file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)

            # Create an invalid YAML file
            with open(library_dir / "invalid.yaml", "w") as f:
                f.write("invalid: yaml: content: [")

            # Create a valid material file
            material_data = {
                "uid": "material1",
                "name": "Material 1",
                "category": "test",
            }
            with open(library_dir / "material1.yaml", "w") as f:
                yaml.dump(material_data, f)

            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            # Should still load the valid material
            assert library.is_loaded is True
            assert len(library) == 1
            assert library.get_material("material1") is not None

    def test_library_get_material(self):
        """Test getting a material from a library."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)

            # Create a material file
            material_data = {
                "uid": "material1",
                "name": "Material 1",
                "category": "test",
            }
            with open(library_dir / "material1.yaml", "w") as f:
                yaml.dump(material_data, f)

            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            material = library.get_material("material1")
            assert material is not None
            assert material.uid == "material1"
            assert material.name == "Material 1"

            # Test getting a non-existent material
            assert library.get_material("nonexistent") is None

    def test_library_add_material(self):
        """Test adding a material to a library."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)
            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            # Create a material
            material = Material(
                uid="material1", name="Material 1", category="test"
            )

            # Add the material
            result = library.add_material(material)

            assert result is True
            assert len(library) == 1
            assert library.get_material("material1") is not None

            # Verify the file was created
            assert (library_dir / "material1.yaml").exists()

    def test_library_add_duplicate_material(self):
        """Test adding a duplicate material to a library."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)
            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            # Create a material
            material = Material(
                uid="material1", name="Material 1", category="test"
            )

            # Add the material twice
            result1 = library.add_material(material)
            result2 = library.add_material(material)

            assert result1 is True
            assert result2 is False
            assert len(library) == 1

    def test_library_add_read_only(self):
        """Test adding a material to a read-only library."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)
            library = MaterialLibrary(library_dir, source="core")
            library.load_materials()

            # Create a material
            material = Material(
                uid="material1", name="Material 1", category="test"
            )

            # Try to add the material
            result = library.add_material(material)

            assert result is False
            assert len(library) == 0

    def test_library_remove_material(self):
        """Test removing a material from a library."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)
            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            # Create a material
            material = Material(
                uid="material1", name="Material 1", category="test"
            )

            # Add the material
            library.add_material(material)
            assert len(library) == 1

            # Remove the material
            result = library.remove_material("material1")

            assert result is True
            assert len(library) == 0
            assert library.get_material("material1") is None

            # Verify the file was removed
            assert not (library_dir / "material1.yaml").exists()

    def test_library_remove_nonexistent_material(self):
        """Test removing a non-existent material from a library."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)
            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            # Try to remove a non-existent material
            result = library.remove_material("nonexistent")

            assert result is False

    def test_library_remove_read_only(self):
        """Test removing a material from a read-only library."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)
            library = MaterialLibrary(library_dir, source="core")
            library.load_materials()

            # Try to remove a material
            result = library.remove_material("material1")

            assert result is False

    def test_library_reload(self):
        """Test reloading a library."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)
            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            # Initially empty
            assert len(library) == 0

            # Add a material file directly
            material_data = {
                "uid": "material1",
                "name": "Material 1",
                "category": "test",
            }
            with open(library_dir / "material1.yaml", "w") as f:
                yaml.dump(material_data, f)

            # Reload the library
            library.reload()

            # Should now have the material
            assert len(library) == 1
            assert library.get_material("material1") is not None

    def test_library_contains(self):
        """Test checking if a library contains a material."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)
            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            # Create a material
            material = Material(
                uid="material1", name="Material 1", category="test"
            )

            # Initially not contained
            assert "material1" not in library

            # Add the material
            library.add_material(material)

            # Now contained
            assert "material1" in library
            assert "nonexistent" not in library

    def test_library_iteration(self):
        """Test iterating over materials in a library."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)
            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            # Create materials
            material1 = Material(uid="material1", name="Material 1")
            material2 = Material(uid="material2", name="Material 2")

            # Add materials
            library.add_material(material1)
            library.add_material(material2)

            # Iterate over materials
            uids = []
            for material in library:
                uids.append(material.uid)

            assert "material1" in uids
            assert "material2" in uids

    def test_library_str_repr(self):
        """Test string representations of MaterialLibrary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)
            library = MaterialLibrary(library_dir, source="user")

            assert "MaterialLibrary" in str(library)
            assert library_dir.name in str(library)
            assert "read_only=False" in str(library)
            assert "materials=0" in str(library)

    def test_library_load_with_metadata(self):
        """Test loading a library with __library__.yaml metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)

            # Create __library__.yaml with metadata
            metadata = {"name": "Test Library", "id": "test-library-uuid-1234"}
            with open(library_dir / "__library__.yaml", "w") as f:
                yaml.dump(metadata, f)

            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            assert library.is_loaded is True
            assert library.display_name == "Test Library"
            assert library.library_id == "test-library-uuid-1234"

    def test_library_load_with_partial_metadata(self):
        """Test loading a library with partial __library__.yaml metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)

            # Create __library__.yaml with only name
            metadata = {"name": "Test Library"}
            with open(library_dir / "__library__.yaml", "w") as f:
                yaml.dump(metadata, f)

            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            assert library.is_loaded is True
            assert library.display_name == "Test Library"
            assert (
                library.library_id == library_dir.name
            )  # Falls back to directory name

    def test_library_load_with_invalid_metadata(self):
        """Test loading a library with invalid __library__.yaml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)

            # Create invalid __library__.yaml
            with open(library_dir / "__library__.yaml", "w") as f:
                f.write("invalid: yaml: content:[")

            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            # Should still load but with fallback values
            assert library.is_loaded is True
            assert (
                library.display_name == library_dir.name
            )  # Falls back to directory name
            assert (
                library.library_id == library_dir.name
            )  # Falls back to directory name

    def test_library_id_fallback_to_directory_name(self):
        """
        Test library_id falls back to directory name when not in metadata.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir) / "test_dir"
            library_dir.mkdir()

            # No __library__.yaml file
            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            assert library.is_loaded is True
            assert library.library_id == "test_dir"
            assert library.display_name == "test_dir"

    def test_library_display_name_fallback_to_directory_name(self):
        """
        Test display_name falls back to directory name when not in metadata.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir) / "test_dir"
            library_dir.mkdir()

            # Create __library__.yaml with only id
            metadata = {"id": "test-uuid"}
            with open(library_dir / "__library__.yaml", "w") as f:
                yaml.dump(metadata, f)

            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            assert library.is_loaded is True
            assert library.library_id == "test-uuid"
            assert (
                library.display_name == "test_dir"
            )  # Falls back to directory name

    def test_library_metadata_with_materials(self):
        """Test loading a library with both metadata and materials."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)

            # Create __library__.yaml with metadata
            metadata = {"name": "Test Library", "id": "test-library-uuid-5678"}
            with open(library_dir / "__library__.yaml", "w") as f:
                yaml.dump(metadata, f)

            # Create a material file
            material_data = {
                "uid": "material1",
                "name": "Material 1",
                "category": "test",
            }
            with open(library_dir / "material1.yaml", "w") as f:
                yaml.dump(material_data, f)

            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            assert library.is_loaded is True
            assert library.display_name == "Test Library"
            assert library.library_id == "test-library-uuid-5678"
            assert len(library) == 1
            assert library.get_material("material1") is not None

    def test_library_set_display_name(self):
        """Test setting the display name of a library."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir)

            # Create __library__.yaml with metadata
            metadata = {
                "name": "Original Library",
                "id": "test-library-uuid-1234",
            }
            with open(library_dir / "__library__.yaml", "w") as f:
                yaml.dump(metadata, f)

            library = MaterialLibrary(library_dir, source="user")
            library.load_materials()

            assert library.display_name == "Original Library"

            # Set new display name
            library.set_display_name("New Library Name")

            # Check that the display name was updated in memory
            assert library._display_name == "New Library Name"
            assert library.display_name == "New Library Name"

            # Verify the library_id remains unchanged
            assert library.library_id == "test-library-uuid-1234"

    def test_library_create(self):
        """Test creating a new library using the class method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            library_dir = Path(temp_dir) / "new_library"

            # Create a new library
            library = MaterialLibrary.create(library_dir, "New Test Library")

            assert library is not None
            assert library.display_name == "New Test Library"
            assert library.library_id is not None
            assert library.source == "user"
            assert library._directory == library_dir

            # Verify the metadata file was created
            meta_file = library_dir / "__library__.yaml"
            assert meta_file.exists()

            # Verify the metadata content
            import yaml

            with open(meta_file, "r") as f:
                metadata = yaml.safe_load(f)
            assert metadata["name"] == "New Test Library"
            assert metadata["id"] == library.library_id

            # Test creating with empty name (should fail)
            failed_library = MaterialLibrary.create(library_dir, "")
            assert failed_library is None

            # Test creating in existing directory (should fail)
            existing_library = MaterialLibrary.create(
                library_dir, "Another Library"
            )
            assert existing_library is None
