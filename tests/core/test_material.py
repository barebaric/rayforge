"""Tests for the Material class."""

import pytest
import tempfile
import yaml
from pathlib import Path
from rayforge.core.material import Material, MaterialAppearance


class TestMaterial:
    """Test cases for the Material class."""

    def test_material_creation(self):
        """Test creating a Material with basic properties."""
        appearance = MaterialAppearance(color="#FF0000", pattern="solid")
        material = Material(
            uid="test_material",
            name="Test Material",
            description="A test material",
            category="test",
            appearance=appearance,
        )

        assert material.uid == "test_material"
        assert material.name == "Test Material"
        assert material.description == "A test material"
        assert material.category == "test"
        assert material.appearance.color == "#FF0000"
        assert material.appearance.pattern == "solid"

    def test_material_default_name(self):
        """Test that material uses UID as name when name is empty."""
        material = Material(uid="test_uid")

        assert material.name == "test_uid"

    def test_material_to_dict(self):
        """Test converting a Material to a dictionary."""
        appearance = MaterialAppearance(color="#FF0000", pattern="solid")
        material = Material(
            uid="test_material",
            name="Test Material",
            description="A test material",
            category="test",
            appearance=appearance,
        )

        data = material.to_dict()

        assert data["uid"] == "test_material"
        assert data["name"] == "Test Material"
        assert data["description"] == "A test material"
        assert data["category"] == "test"
        assert data["appearance"]["color"] == "#FF0000"
        assert data["appearance"]["pattern"] == "solid"

    def test_material_from_file(self):
        """Test loading a Material from a YAML file."""
        material_data = {
            "uid": "test_material",
            "name": "Test Material",
            "description": "A test material",
            "category": "test",
            "appearance": {"color": "#FF0000", "pattern": "solid"},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(material_data, f)
            file_path = Path(f.name)

        try:
            material = Material.from_file(file_path)

            assert material.uid == "test_material"
            assert material.name == "Test Material"
            assert material.description == "A test material"
            assert material.category == "test"
            assert material.appearance.color == "#FF0000"
            assert material.appearance.pattern == "solid"
            assert material.file_path == file_path
        finally:
            file_path.unlink()

    def test_material_from_missing_file(self):
        """Test loading a Material from a non-existent file."""
        with pytest.raises(FileNotFoundError):
            Material.from_file(Path("non_existent_file.yaml"))

    def test_material_from_invalid_yaml(self):
        """Test loading a Material from a file with invalid YAML."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: content: [")
            file_path = Path(f.name)

        try:
            with pytest.raises(yaml.YAMLError):
                Material.from_file(file_path)
        finally:
            file_path.unlink()

    def test_material_from_non_dict_file(self):
        """
        Test loading a Material from a file that doesn't contain a dictionary.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write('"not a dictionary"')
            file_path = Path(f.name)

        try:
            with pytest.raises(ValueError):
                Material.from_file(file_path)
        finally:
            file_path.unlink()

    def test_material_save_to_file(self):
        """Test saving a Material to a YAML file."""
        appearance = MaterialAppearance(color="#FF0000", pattern="solid")
        material = Material(
            uid="test_material",
            name="Test Material",
            description="A test material",
            category="test",
            appearance=appearance,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_material.yaml"

            material.save_to_file(file_path)

            assert file_path.exists()
            assert material.file_path == file_path

            # Verify the content
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)

            assert data["uid"] == "test_material"
            assert data["name"] == "Test Material"
            assert data["description"] == "A test material"
            assert data["category"] == "test"
            assert data["appearance"]["color"] == "#FF0000"
            assert data["appearance"]["pattern"] == "solid"

    def test_material_save_to_existing_file(self):
        """Test saving a Material to an existing file."""
        appearance = MaterialAppearance(color="#FF0000", pattern="solid")
        material = Material(
            uid="test_material",
            name="Test Material",
            description="A test material",
            category="test",
            appearance=appearance,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_material.yaml"

            # Create the file first
            file_path.touch()

            material.save_to_file(file_path)

            assert file_path.exists()
            assert material.file_path == file_path

    def test_material_save_without_file_path(self):
        """Test saving a Material without specifying a file path."""
        material = Material(
            uid="test_material",
            name="Test Material",
            description="A test material",
            category="test",
        )

        with pytest.raises(ValueError):
            material.save_to_file()

    def test_material_get_display_color(self):
        """Test getting the display color of a Material."""
        appearance = MaterialAppearance(color="#FF0000", pattern="solid")
        material = Material(
            uid="test_material",
            appearance=appearance,
        )

        assert material.get_display_color() == "#FF0000"

        # Test default color when not specified
        material_no_color = Material(uid="test_material")
        assert material_no_color.get_display_color() == "#f0f0f0"

    def test_material_get_pattern(self):
        """Test getting the pattern of a Material."""
        appearance = MaterialAppearance(color="#FF0000", pattern="wood_grain")
        material = Material(
            uid="test_material",
            appearance=appearance,
        )

        assert material.get_pattern() == "wood_grain"

        # Test default pattern when not specified
        material_no_pattern = Material(uid="test_material")
        assert material_no_pattern.get_pattern() == "solid"

    def test_material_str_repr(self):
        """Test string representations of Material."""
        material = Material(
            uid="test_material",
            name="Test Material",
            description="A test material",
            category="test",
        )

        assert str(material) == (
            "Material(uid='test_material', name='Test Material')"
        )
        assert "Material(uid='test_material', name='Test Material'" in repr(
            material
        )
        assert "category='test'" in repr(material)
        assert "description='A test material'" in repr(material)


class TestMaterialAppearance:
    """Test cases for the MaterialAppearance class."""

    def test_appearance_creation(self):
        """Test creating a MaterialAppearance with basic properties."""
        appearance = MaterialAppearance(color="#FF0000", pattern="solid")

        assert appearance.color == "#FF0000"
        assert appearance.pattern == "solid"

    def test_appearance_defaults(self):
        """Test MaterialAppearance default values."""
        appearance = MaterialAppearance()

        assert appearance.color == "#f0f0f0"
        assert appearance.pattern == "solid"

    def test_appearance_from_dict(self):
        """Test creating a MaterialAppearance from a dictionary."""
        data = {"color": "#00FF00", "pattern": "wood_grain"}
        appearance = MaterialAppearance.from_dict(data)

        assert appearance.color == "#00FF00"
        assert appearance.pattern == "wood_grain"

    def test_appearance_from_dict_defaults(self):
        """Test creating a MaterialAppearance from a dict with missing keys."""
        data = {}
        appearance = MaterialAppearance.from_dict(data)

        assert appearance.color == "#f0f0f0"
        assert appearance.pattern == "solid"

    def test_appearance_from_dict_partial(self):
        """Test creating a MaterialAppearance from a dict with partial data."""
        data = {"color": "#0000FF"}
        appearance = MaterialAppearance.from_dict(data)

        assert appearance.color == "#0000FF"
        assert appearance.pattern == "solid"

    def test_appearance_to_dict(self):
        """Test converting a MaterialAppearance to a dictionary."""
        appearance = MaterialAppearance(color="#FF0000", pattern="solid")
        data = appearance.to_dict()

        assert data["color"] == "#FF0000"
        assert data["pattern"] == "solid"

    def test_material_get_display_rgba(self):
        """Test getting the display color as RGBA tuple."""
        appearance = MaterialAppearance(color="#FF0000", pattern="solid")
        material = Material(
            uid="test_material",
            appearance=appearance,
        )

        rgba = material.get_display_rgba()
        assert rgba == (1.0, 0.0, 0.0, 1.0)

    def test_material_get_display_rgba_with_alpha(self):
        """Test getting the display color as RGBA tuple with custom alpha."""
        appearance = MaterialAppearance(color="#00FF00", pattern="solid")
        material = Material(
            uid="test_material",
            appearance=appearance,
        )

        rgba = material.get_display_rgba(alpha=0.5)
        assert rgba == (0.0, 1.0, 0.0, 0.5)

    def test_material_get_display_rgba_fallback(self):
        """Test getting RGBA with invalid color format."""
        appearance = MaterialAppearance(color="invalid_color", pattern="solid")
        material = Material(
            uid="test_material",
            appearance=appearance,
        )

        rgba = material.get_display_rgba()
        assert rgba == (0.5, 0.5, 0.5, 1.0)

    def test_material_get_display_rgba_no_hash(self):
        """Test getting RGBA with color without # prefix."""
        appearance = MaterialAppearance(color="0000FF", pattern="solid")
        material = Material(
            uid="test_material",
            appearance=appearance,
        )

        rgba = material.get_display_rgba()
        assert rgba == (0.0, 0.0, 1.0, 1.0)

    def test_material_get_display_rgba_lowercase(self):
        """Test getting RGBA with lowercase hex color."""
        appearance = MaterialAppearance(color="#ff00ff", pattern="solid")
        material = Material(
            uid="test_material",
            appearance=appearance,
        )

        rgba = material.get_display_rgba()
        assert rgba == (1.0, 0.0, 1.0, 1.0)
