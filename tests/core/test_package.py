import tempfile
import yaml
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
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
            "depends": ["rayforge>=0.27.0,~0.27"],
            "author": {"name": "Test User", "email": "test@example.com"},
            "repository": "https://github.com/example/repo",
            "version": "1.0.0",
        }
        meta = PackageMetadata.from_registry_entry("my_plugin", data)
        assert meta.name == "my_plugin"
        assert meta.display_name == "My Plugin"
        assert meta.author.name == "Test User"
        assert meta.url == "https://github.com/example/repo"
        assert meta.depends == ["rayforge>=0.27.0,~0.27"]

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
                "author": {"name": "Me", "email": "me@me.com"},
                "entry_point": "main.py",
            }
            with open(path / "rayforge-package.yaml", "w") as f:
                yaml.dump(data, f)

            pkg = Package.load_from_directory(path)
            # The package name must be the directory name, not from the file.
            assert pkg.metadata.name == path.name
            # Version comes from git tags (defaults to 0.0.1 if no tags)
            assert pkg.metadata.version == "0.0.1"
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
                "depends": ["rayforge>=0.27.0,~0.27"],
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
                "depends": ["rayforge>=0.27.0,~0.27"],
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
                "depends": ["rayforge>=0.27.0,~0.27"],
                "author": {"name": "Bad", "email": "bad@bad.com"},
                "provides": {"assets": [{"path": "../../../etc/passwd"}]},
            }
            with open(path / "rayforge-package.yaml", "w") as f:
                yaml.dump(data, f)

            pkg = Package.load_from_directory(path)
            with pytest.raises(PackageValidationError) as exc:
                pkg.validate()
            assert "Invalid asset path" in str(exc.value)

    def test_validate_missing_depends(self):
        """Test validation fails if depends is missing."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "main.py").write_text("def my_plugin(): pass")
            data = {
                "name": "test_pkg",
                "author": {"name": "Me", "email": "me@example.com"},
                "provides": {"code": "main.py:my_plugin"},
            }
            with open(path / "rayforge-package.yaml", "w") as f:
                yaml.dump(data, f)

            pkg = Package.load_from_directory(path)
            with pytest.raises(PackageValidationError) as exc:
                pkg.validate()
            assert "depends is required" in str(exc.value)

    def test_validate_invalid_depends(self):
        """Test validation fails if depends has invalid version."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "main.py").write_text("def my_plugin(): pass")
            data = {
                "name": "test_pkg",
                "depends": ["rayforge,>=not-a-version"],
                "author": {"name": "Me", "email": "me@example.com"},
                "provides": {"code": "main.py:my_plugin"},
            }
            with open(path / "rayforge-package.yaml", "w") as f:
                yaml.dump(data, f)

            pkg = Package.load_from_directory(path)
            with pytest.raises(PackageValidationError) as exc:
                pkg.validate()
            assert "Invalid semantic version" in str(exc.value)


class TestGetGitTagVersion:
    def test_returns_default_when_gitpython_not_installed(self):
        """Test returns default version when GitPython is not installed."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with patch("importlib.import_module", side_effect=ImportError):
                result = Package.get_git_tag_version(path)
                assert result == "0.0.1"

    def test_returns_default_when_not_git_repository(self):
        """Test returns default version when directory is not a git repo."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            result = Package.get_git_tag_version(path)
            assert result == "0.0.1"

    def test_returns_default_when_no_tags(self):
        """Test returns default version when git repo has no tags."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            mock_repo = MagicMock()
            mock_repo.tags = []
            mock_repo_class = MagicMock(return_value=mock_repo)

            with patch("git.Repo", mock_repo_class):
                result = Package.get_git_tag_version(path)
                assert result == "0.0.1"
                mock_repo_class.assert_called_once_with(path)

    def test_returns_latest_tag_name(self):
        """Test returns the name of the latest git tag by commit datetime."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)

            now = datetime.now(timezone.utc)
            earlier = now.replace(hour=now.hour - 1)

            tag1 = Mock()
            tag1.name = "v1.0.0"
            tag1.commit.committed_datetime = earlier

            tag2 = Mock()
            tag2.name = "v2.0.0"
            tag2.commit.committed_datetime = now

            mock_repo = MagicMock()
            mock_repo.tags = [tag1, tag2]
            mock_repo_class = MagicMock(return_value=mock_repo)

            with patch("git.Repo", mock_repo_class):
                result = Package.get_git_tag_version(path)
                assert result == "v2.0.0"

    def test_returns_latest_tag_unordered(self):
        """Test correctly orders tags regardless of list order."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)

            now = datetime.now(timezone.utc)
            earlier = now.replace(hour=now.hour - 1)
            earliest = now.replace(hour=now.hour - 2)

            tag1 = Mock()
            tag1.name = "v3.0.0"
            tag1.commit.committed_datetime = now

            tag2 = Mock()
            tag2.name = "v1.0.0"
            tag2.commit.committed_datetime = earliest

            tag3 = Mock()
            tag3.name = "v2.0.0"
            tag3.commit.committed_datetime = earlier

            mock_repo = MagicMock()
            mock_repo.tags = [tag2, tag3, tag1]
            mock_repo_class = MagicMock(return_value=mock_repo)

            with patch("git.Repo", mock_repo_class):
                result = Package.get_git_tag_version(path)
                assert result == "v3.0.0"

    def test_handles_exception_gracefully(self):
        """Test returns default version when an exception occurs."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            mock_repo_class = MagicMock(side_effect=Exception("Git error"))

            with patch("git.Repo", mock_repo_class):
                result = Package.get_git_tag_version(path)
                assert result == "0.0.1"
