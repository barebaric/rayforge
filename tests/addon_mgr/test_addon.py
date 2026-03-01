import sys
import tempfile
import yaml
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from rayforge.addon_mgr.addon import (
    Addon,
    AddonMetadata,
    AddonValidationError,
    AddonAuthor,
    AddonProvides,
)
from rayforge.shared.util.versioning import get_git_tag_version, UnknownVersion


TEST_VERSION = "1.0.0"


class TestAddonMetadata:
    def test_from_registry_entry_valid(self):
        """Test creating metadata from registry dictionary."""
        data = {
            "name": "My Addon",
            "description": "A test addon",
            "depends": ["rayforge>=0.27.0,~0.27"],
            "author": {"name": "Test User", "email": "test@example.com"},
            "repository": "https://github.com/example/repo",
            "version": "1.0.0",
        }
        meta = AddonMetadata.from_registry_entry("my_addon", data)
        assert meta.name == "my_addon"
        assert meta.display_name == "My Addon"
        assert meta.author.name == "Test User"
        assert meta.url == "https://github.com/example/repo"
        assert meta.depends == ["rayforge>=0.27.0,~0.27"]

    def test_from_registry_entry_with_string_depends(self):
        """Test handling string depends field."""
        data = {
            "name": "My Addon",
            "depends": "rayforge>=0.27.0",
            "author": "Test User <test@example.com>",
            "version": "1.0.0",
        }
        meta = AddonMetadata.from_registry_entry("my_addon", data)
        assert meta.depends == ["rayforge>=0.27.0"]
        assert meta.author.name == "Test User"
        assert meta.author.email == "test@example.com"

    def test_from_registry_entry_with_string_author(self):
        """Test handling string author field without email."""
        data = {
            "name": "My Addon",
            "depends": [],
            "author": "Test User",
            "version": "1.0.0",
        }
        meta = AddonMetadata.from_registry_entry("my_addon", data)
        assert meta.author.name == "Test User"
        assert meta.author.email == ""

    def test_from_registry_entry_missing_optional_fields(self):
        """Test handling missing optional fields."""
        data = {"depends": []}
        meta = AddonMetadata.from_registry_entry("my_addon", data)
        assert meta.description == ""
        assert meta.author.name == ""
        assert meta.author.email == ""

    def test_to_dict(self):
        """Test converting metadata to dictionary."""
        meta = AddonMetadata(
            name="test",
            description="Test addon",
            version="1.0.0",
            depends=["rayforge>=0.27.0"],
            author=AddonAuthor(name="Test", email="test@test.com"),
            provides=AddonProvides(),
        )
        d = meta.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "1.0.0"

    def test_to_dict_with_unknown_version(self):
        """Test converting metadata with UnknownVersion to dictionary."""
        meta = AddonMetadata(
            name="test",
            description="Test addon",
            version=UnknownVersion,
            depends=["rayforge>=0.27.0"],
            author=AddonAuthor(name="Test", email="test@test.com"),
            provides=AddonProvides(),
        )
        d = meta.to_dict()
        assert d["version"] is None


class TestAddon:
    def test_load_from_directory_success(self):
        """Test loading an addon from a directory."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            data = {
                "name": "test_pkg",
                "author": {"name": "Me", "email": "me@me.com"},
                "provides": {"backend": "main.py"},
            }
            with open(path / "rayforge-addon.yaml", "w") as f:
                yaml.dump(data, f)

            addon = Addon.load_from_directory(path, version=TEST_VERSION)
            assert addon.metadata.name == path.name
            assert addon.metadata.version == TEST_VERSION
            assert addon.metadata.provides.backend == "main.py"

    def test_load_missing_metadata(self):
        """Test error when metadata file is missing."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError):
                Addon.load_from_directory(Path(tmp), version=TEST_VERSION)

    def test_validate_success_simple(self):
        """Test basic validation passes."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "main.py").write_text("def my_plugin(): pass")

            data = {
                "name": "test_pkg",
                "depends": ["rayforge>=0.27.0,~0.27"],
                "author": {"name": "Me", "email": "me@example.com"},
                "provides": {"backend": "main"},
            }
            with open(path / "rayforge-addon.yaml", "w") as f:
                yaml.dump(data, f)

            addon = Addon.load_from_directory(path, version=TEST_VERSION)
            assert addon.validate() is True

    def test_validate_rejects_invalid_module_path(self):
        """Test validation rejects invalid module paths."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "main.py").write_text("def my_plugin(): pass")

            data = {
                "name": "test_pkg",
                "depends": ["rayforge>=0.27.0,~0.27"],
                "author": {"name": "Me", "email": "me@example.com"},
                "provides": {"backend": "main:my_plugin"},
            }
            with open(path / "rayforge-addon.yaml", "w") as f:
                yaml.dump(data, f)

            addon = Addon.load_from_directory(path, version=TEST_VERSION)
            with pytest.raises(AddonValidationError) as exc:
                addon.validate()
            assert "not a valid module path" in str(exc.value)

    def test_validate_missing_module(self):
        """Test validation fails if module is not found."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            data = {
                "name": "test_pkg",
                "depends": ["rayforge>=0.27.0,~0.27"],
                "author": {"name": "Me", "email": "me@example.com"},
                "provides": {"backend": "nonexistent"},
            }
            with open(path / "rayforge-addon.yaml", "w") as f:
                yaml.dump(data, f)

            addon = Addon.load_from_directory(path, version=TEST_VERSION)
            with pytest.raises(AddonValidationError) as exc:
                addon.validate()
            assert "not found" in str(exc.value)

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
            with open(path / "rayforge-addon.yaml", "w") as f:
                yaml.dump(data, f)

            addon = Addon.load_from_directory(path, version=TEST_VERSION)
            with pytest.raises(AddonValidationError) as exc:
                addon.validate()
            assert "Invalid asset path" in str(exc.value)

    def test_validate_missing_depends(self):
        """Test validation fails if depends is missing."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "main.py").write_text("def my_plugin(): pass")
            data = {
                "name": "test_pkg",
                "author": {"name": "Me", "email": "me@example.com"},
                "provides": {"backend": "main"},
            }
            with open(path / "rayforge-addon.yaml", "w") as f:
                yaml.dump(data, f)

            addon = Addon.load_from_directory(path, version=TEST_VERSION)
            with pytest.raises(AddonValidationError) as exc:
                addon.validate()
            assert "depends is required" in str(exc.value)

    def test_validate_invalid_depends(self):
        """Test validation fails for invalid depends format."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "main.py").write_text("def my_plugin(): pass")
            data = {
                "name": "test_pkg",
                "depends": ["rayforge>=0.27.0,", ",test"],
                "author": {"name": "Me", "email": "me@example.com"},
                "provides": {"backend": "main"},
            }
            with open(path / "rayforge-addon.yaml", "w") as f:
                yaml.dump(data, f)

            addon = Addon.load_from_directory(path, version=TEST_VERSION)
            with pytest.raises(AddonValidationError):
                addon.validate()

    def test_validate_wrong_api_version(self):
        """Test validation fails for wrong api_version."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "main.py").write_text("def my_plugin(): pass")
            data = {
                "name": "test_pkg",
                "depends": ["rayforge>=0.27.0,~0.27"],
                "author": {"name": "Me", "email": "me@example.com"},
                "provides": {"backend": "main"},
                "api_version": 999,
            }
            with open(path / "rayforge-addon.yaml", "w") as f:
                yaml.dump(data, f)

            addon = Addon.load_from_directory(path, version=TEST_VERSION)
            with pytest.raises(AddonValidationError) as exc:
                addon.validate()
            assert "Unsupported api_version" in str(exc.value)

    def test_validate_api_version_not_integer(self):
        """Test validation fails if api_version is not an integer."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "main.py").write_text("def my_plugin(): pass")
            data = {
                "name": "test_pkg",
                "depends": ["rayforge>=0.27.0,~0.27"],
                "author": {"name": "Me", "email": "me@example.com"},
                "provides": {"backend": "main"},
                "api_version": "1",
            }
            with open(path / "rayforge-addon.yaml", "w") as f:
                yaml.dump(data, f)

            addon = Addon.load_from_directory(path, version=TEST_VERSION)
            with pytest.raises(AddonValidationError) as exc:
                addon.validate()
            assert "must be an integer" in str(exc.value)

    def test_validate_default_api_version(self):
        """Test default api_version is used if not specified."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "main.py").write_text("def my_plugin(): pass")
            data = {
                "name": "test_pkg",
                "depends": ["rayforge>=0.27.0,~0.27"],
                "author": {"name": "Me", "email": "me@example.com"},
                "provides": {"backend": "main"},
            }
            with open(path / "rayforge-addon.yaml", "w") as f:
                yaml.dump(data, f)

            addon = Addon.load_from_directory(path, version=TEST_VERSION)
            assert addon.validate() is True
            assert addon.metadata.api_version == 1

    def test_validate_with_unknown_version(self):
        """Test validation passes with UnknownVersion."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "main.py").write_text("def my_plugin(): pass")
            data = {
                "name": "test_pkg",
                "depends": ["rayforge>=0.27.0,~0.27"],
                "author": {"name": "Me", "email": "me@example.com"},
                "provides": {"backend": "main"},
            }
            with open(path / "rayforge-addon.yaml", "w") as f:
                yaml.dump(data, f)

            addon = Addon.load_from_directory(path, version=UnknownVersion)
            assert addon.validate() is True
            assert addon.metadata.version is UnknownVersion


class TestGetGitTagVersion:
    def test_raises_when_gitpython_not_installed(self):
        """Test raises RuntimeError when GitPython is not installed."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with patch.dict(sys.modules, {"git": None, "git.Repo": None}):
                with patch(
                    "rayforge.shared.util.versioning.importlib.import_module",
                    side_effect=ImportError,
                ):
                    with pytest.raises(RuntimeError, match="GitPython"):
                        get_git_tag_version(path)

    def test_raises_when_not_git_repository(self):
        """Test raises RuntimeError when directory is not a git repo."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            mock_git = MagicMock()
            mock_git.Repo.side_effect = Exception("Not a git repo")

            with patch.dict(sys.modules, {"git": mock_git}):
                with pytest.raises(RuntimeError, match="Failed to get"):
                    get_git_tag_version(path)

    def test_raises_when_no_tags(self):
        """Test raises RuntimeError when git repo has no tags."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            mock_repo = MagicMock()
            mock_repo.tags = []
            mock_git = MagicMock()
            mock_git.Repo.return_value = mock_repo

            with patch.dict(sys.modules, {"git": mock_git}):
                with pytest.raises(RuntimeError, match="No git tags found"):
                    get_git_tag_version(path)

    def test_returns_latest_tag_name(self):
        """Test returns the name of the latest git tag by commit datetime."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)

            now = datetime.now(timezone.utc)
            earlier = now - timedelta(hours=1)

            tag1 = Mock()
            tag1.name = "v1.0.0"
            tag1.commit.committed_datetime = earlier

            tag2 = Mock()
            tag2.name = "v2.0.0"
            tag2.commit.committed_datetime = now

            mock_repo = MagicMock()
            mock_repo.tags = [tag1, tag2]
            mock_git = MagicMock()
            mock_git.Repo.return_value = mock_repo

            with patch.dict(sys.modules, {"git": mock_git}):
                result = get_git_tag_version(path)
                assert result == "v2.0.0"

    def test_returns_latest_tag_unordered(self):
        """Test correctly orders tags regardless of list order."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)

            now = datetime.now(timezone.utc)
            earlier = now - timedelta(hours=1)
            earliest = now - timedelta(hours=2)

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
            mock_git = MagicMock()
            mock_git.Repo.return_value = mock_repo

            with patch.dict(sys.modules, {"git": mock_git}):
                result = get_git_tag_version(path)
                assert result == "v3.0.0"

    def test_raises_on_exception(self):
        """Test raises RuntimeError when an exception occurs."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            mock_git = MagicMock()
            mock_git.Repo.side_effect = Exception("Git error")

            with patch.dict(sys.modules, {"git": mock_git}):
                with pytest.raises(RuntimeError, match="Git error"):
                    get_git_tag_version(path)
