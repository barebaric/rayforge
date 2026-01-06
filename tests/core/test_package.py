import tempfile
import yaml
import pytest
from pathlib import Path
from rayforge.core.package import (
    Package,
    PackageMetadata,
    PackageValidationError,
)


class TestPackageMetadata:
    def test_from_registry_entry_valid(self):
        """Test creating metadata from registry dictionary."""
        data = {
            "name": "My Plugin",
            "description": "A test plugin",
            "author": {"name": "Test User", "email": "test@example.com"},
            "repository": "https://github.com/example/repo",
            "version": "1.0.0",
        }
        meta = PackageMetadata.from_registry_entry("my_plugin", data)
        assert meta.name == "my_plugin"
        assert meta.display_name == "My Plugin"
        assert meta.author.name == "Test User"
        assert meta.url == "https://github.com/example/repo"

    def test_from_registry_entry_string_author(self):
        """Test parsing string format author."""
        data = {"author": "John Doe <john@doe.com>", "name": "Display Name"}
        meta = PackageMetadata.from_registry_entry("pkg_id", data)
        assert meta.author.name == "John Doe"
        assert meta.author.email == "john@doe.com"


class TestPackage:
    def test_load_from_directory_success(self):
        """Test loading a package from a directory."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            data = {
                "name": "test_pkg",  # This name should be ignored
                "version": "0.1.0",
                "author": {"name": "Me", "email": "me@me.com"},
                "entry_point": "main.py",
            }
            with open(path / "rayforge-package.yaml", "w") as f:
                yaml.dump(data, f)

            pkg = Package.load_from_directory(path)
            # The package name must be the directory name, not from the file.
            assert pkg.metadata.name == path.name
            assert pkg.metadata.provides.code == "main.py"

    def test_load_missing_metadata(self):
        """Test error when metadata file is missing."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                Package.load_from_directory(Path(tmp))

    def test_validate_success_simple(self):
        """Test basic validation passes."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            # Create a dummy script
            (path / "main.py").write_text("def my_plugin(): pass")

            data = {
                "name": "test_pkg",
                "author": {"name": "Me", "email": "me@example.com"},
                "provides": {"code": "main.py:my_plugin"},
            }
            with open(path / "rayforge-package.yaml", "w") as f:
                yaml.dump(data, f)

            pkg = Package.load_from_directory(path)
            assert pkg.validate() is True

    def test_validate_fails_ast_check(self):
        """Test validation fails if entry point function is missing."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "main.py").write_text("def other_function(): pass")

            data = {
                "name": "test_pkg",
                "author": {"name": "Me", "email": "me@example.com"},
                "provides": {"code": "main.py:missing_func"},
            }
            with open(path / "rayforge-package.yaml", "w") as f:
                yaml.dump(data, f)

            pkg = Package.load_from_directory(path)
            with pytest.raises(PackageValidationError) as exc:
                pkg.validate()
            assert "Attribute 'missing_func' not found" in str(exc.value)

    def test_validate_path_traversal(self):
        """Test asset path security check."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            data = {
                "name": "hack_pkg",
                "author": {"name": "Bad", "email": "bad@bad.com"},
                "provides": {"assets": [{"path": "../../../etc/passwd"}]},
            }
            with open(path / "rayforge-package.yaml", "w") as f:
                yaml.dump(data, f)

            pkg = Package.load_from_directory(path)
            with pytest.raises(PackageValidationError) as exc:
                pkg.validate()
            assert "Invalid asset path" in str(exc.value)
