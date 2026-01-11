import ast
import importlib
import logging
import re
import yaml
import semver
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

METADATA_FILENAME = "rayforge-package.yaml"
LEGACY_METADATA_FILENAME = "rayforge_package.yaml"


class PackageValidationError(Exception):
    """
    Raised when a package fails validation checks.
    """

    pass


@dataclass
class PackageAuthor:
    """Represents the author of a package."""

    name: str
    email: str


@dataclass
class PackageProvides:
    """
    Defines what the package provides to the system.

    Attributes:
        code (Optional[str]): The entry point for the package
                              (e.g., 'module.submodule:Class').
        assets (List[Dict[str, str]]): A list of asset definitions.
    """

    code: Optional[str] = None
    assets: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PackageMetadata:
    """
    Serializable metadata for a Rayforge package.
    """

    name: str
    description: str
    version: str
    depends: List[str]
    author: PackageAuthor
    provides: PackageProvides
    url: str = ""
    display_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Converts metadata back to a dictionary for YAML serialization."""
        return asdict(self)

    @classmethod
    def from_registry_entry(
        cls, pkg_id: str, data: Dict[str, Any]
    ) -> "PackageMetadata":
        """
        Parses a registry dictionary entry into a PackageMetadata object.
        Handles normalization of author fields and mapping keys.
        """
        # Parse Author
        author_info = data.get("author", {})
        if isinstance(author_info, dict):
            author = PackageAuthor(
                name=author_info.get("name", ""),
                email=author_info.get("email", ""),
            )
        elif isinstance(author_info, str):
            # Handle "Name <email>" string format if present, else just Name
            match = re.match(r"(.*) <(.*)>", author_info)
            if match:
                author = PackageAuthor(
                    name=match.group(1).strip(), email=match.group(2).strip()
                )
            else:
                author = PackageAuthor(name=author_info, email="")
        else:
            author = PackageAuthor(name="Unknown", email="")

        # Registry usually doesn't provide granular 'provides' details
        # We populate basics needed for the UI listing
        provides = PackageProvides()

        depends = data.get("depends", [])
        if isinstance(depends, str):
            depends = [depends]

        return cls(
            name=pkg_id,
            display_name=data.get("name", pkg_id),
            description=data.get("description", ""),
            version=str(
                data.get("latest_stable", data.get("version", "0.0.0"))
            ),
            depends=depends,
            author=author,
            provides=provides,
            url=data.get("repository", ""),
        )


class Package:
    """
    A class representing a loadable Rayforge package.
    """

    def __init__(self, path: Path, metadata: PackageMetadata):
        """
        Initialize the Package.

        Args:
            path (Path): The root directory of the package on the filesystem.
            metadata (PackageMetadata): The parsed metadata object.
        """
        self.root_path = path
        self.metadata = metadata

    @classmethod
    def load_from_directory(cls, package_dir: Path) -> "Package":
        """
        Loads a package from a directory by parsing its YAML metadata file.
        The directory name is treated as the canonical package ID.
        The version is obtained from git tags.

        Args:
            package_dir (Path): The directory containing the package.
        """
        meta_file = package_dir / METADATA_FILENAME
        if not meta_file.exists():
            # Fallback for older legacy naming convention
            meta_file = package_dir / LEGACY_METADATA_FILENAME

        if not meta_file.exists():
            raise FileNotFoundError(
                f"No package metadata file ('{METADATA_FILENAME}') "
                f"found in {package_dir}"
            )

        try:
            with open(meta_file, "r") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise PackageValidationError(f"Failed to parse YAML metadata: {e}")

        try:
            # 1. Parse Author
            author_data = data.get("author", {})
            if isinstance(author_data, str):
                match = re.match(r"(.*) <(.*)>", author_data)
                if match:
                    author = PackageAuthor(
                        name=match.group(1).strip(),
                        email=match.group(2).strip(),
                    )
                else:
                    author = PackageAuthor(name=author_data, email="")
            else:
                author = PackageAuthor(
                    name=author_data.get("name", ""),
                    email=author_data.get("email", ""),
                )

            # 2. Parse 'Provides' (Code and Assets)
            provides_data = data.get("provides", {})

            # Legacy compatibility: if 'entry_point' is at root, map it
            if "code" not in provides_data and "entry_point" in data:
                provides_data["code"] = data["entry_point"]

            provides = PackageProvides(
                code=provides_data.get("code"),
                assets=provides_data.get("assets", []),
            )

            # 3. Construct Metadata
            # The directory name is the source of truth for the package
            # name (ID).
            depends = data.get("depends", [])
            if isinstance(depends, str):
                depends = [depends]

            version = cls.get_git_tag_version(package_dir)

            metadata = PackageMetadata(
                name=package_dir.name,
                display_name=data.get("display_name", data.get("name", "")),
                description=data.get("description", ""),
                version=version,
                depends=depends,
                author=author,
                provides=provides,
                url=data.get("url", ""),
            )

            return cls(path=package_dir, metadata=metadata)

        except Exception as e:
            raise PackageValidationError(f"Structure error in metadata: {e}")

    def validate(self) -> bool:
        """
        Performs rigorous validation on the package.
        """
        logger.debug(f"Validating package structure for: {self.metadata.name}")

        # --- 1. Basic Field Validation ---
        if not self.metadata.name or not self.metadata.name.strip():
            raise PackageValidationError("Package 'name' cannot be empty.")

        if not self.metadata.description:
            logger.warning(
                f"Package '{self.metadata.name}' has no description."
            )

        # --- 2. Version Validation ---
        try:
            # Strip leading 'v' if present for validation
            clean_ver = self.metadata.version.lstrip("v")
            semver.VersionInfo.parse(clean_ver)
        except ValueError:
            raise PackageValidationError(
                f"Invalid semantic version: {self.metadata.version}"
            )

        # --- 2.1 Depends Validation ---
        if not self.metadata.depends:
            raise PackageValidationError("depends is required.")
        for dep in self.metadata.depends:
            if not isinstance(dep, str):
                raise PackageValidationError(
                    f"Dependency must be a string: {dep}"
                )
            parts = dep.split(",")
            if not parts or not parts[0]:
                raise PackageValidationError(
                    f"Invalid dependency format: {dep}"
                )
            pkg_part = parts[0].strip()
            if not pkg_part:
                raise PackageValidationError(
                    f"Invalid dependency format: {dep}"
                )
            # Validate version constraints if present
            for constraint in parts[1:]:
                constraint = constraint.strip()
                if not constraint:
                    continue
                # Extract operator and version
                op_match = re.match(r"^([~^><=!]+)(.+)$", constraint)
                if not op_match:
                    raise PackageValidationError(
                        f"Invalid version constraint '{constraint}' in: {dep}"
                    )
                version_str = op_match.group(2).lstrip("v")
                operator = op_match.group(1)

                # For tilde operator, allow partial versions (e.g., ~0.27)
                # and normalize them to full semver (e.g., ~0.27 -> ~0.27.0)
                if operator == "~":
                    version_parts = version_str.split(".")
                    if len(version_parts) == 2:
                        version_str = f"{version_str}.0"
                    elif len(version_parts) == 1:
                        version_str = f"{version_str}.0.0"

                try:
                    semver.VersionInfo.parse(version_str)
                except ValueError:
                    raise PackageValidationError(
                        f"Invalid semantic version in constraint "
                        f"'{constraint}': {dep}"
                    )

        # --- 3. Author Validation ---
        if not self.metadata.author.name:
            raise PackageValidationError("Author name is required.")

        if "your-github-username" in self.metadata.author.name.lower():
            raise PackageValidationError(
                "Placeholder detected in author name."
            )

        if self.metadata.author.email:
            # Basic regex for email format
            if not re.match(
                r"^[^@\s]+@[^@\s]+\.[^@\s]+$", self.metadata.author.email
            ):
                logger.warning(
                    f"Author email '{self.metadata.author.email}' "
                    "appears invalid."
                )

        # --- 4. Asset Validation ---
        for asset in self.metadata.provides.assets:
            path_str = asset.get("path")
            if not path_str:
                raise PackageValidationError(
                    "Asset entry is missing 'path' key."
                )

            # Security: Prevent path traversal
            if ".." in path_str or path_str.startswith("/"):
                raise PackageValidationError(
                    f"Invalid asset path '{path_str}'. Paths must be relative."
                )

            full_path = self.root_path / path_str
            if not full_path.exists():
                raise PackageValidationError(
                    f"Asset path not found: {path_str}"
                )

        # --- 5. Code Entry Point Validation (Static) ---
        if self.metadata.provides.code:
            self._validate_code_entry_point(self.metadata.provides.code)

        return True

    def _validate_code_entry_point(self, entry_point: str):
        """
        Validates the Python entry point using AST parsing.
        """
        # Case A: Module with Attribute (module.submodule:Class)
        if ":" in entry_point:
            module_str, attr_name = entry_point.split(":", 1)
            file_path = self._resolve_module_path(module_str)
            if not file_path:
                raise PackageValidationError(
                    f"Could not locate module file for '{module_str}' "
                    f"inside {self.root_path}"
                )
            self._check_ast_for_attribute(file_path, attr_name)

        # Case B: Direct file reference (Legacy)
        else:
            file_path = self.root_path / entry_point
            if not file_path.exists():
                raise PackageValidationError(
                    f"Entry point file '{entry_point}' not found."
                )

    def _resolve_module_path(self, module_str: str) -> Optional[Path]:
        """
        Resolves a dotted module string or a filename to a path.
        """
        # Check if the module string is already a direct file path
        direct_file_path = self.root_path / module_str
        if direct_file_path.exists():
            return direct_file_path

        # Convert dots to slashes for package path resolution
        rel_path = module_str.replace(".", "/")

        # Check for directory module (__init__.py)
        path_init = self.root_path / rel_path / "__init__.py"
        if path_init.exists():
            return path_init

        # Check for standard .py file
        path_py = self.root_path / (rel_path + ".py")
        if path_py.exists():
            return path_py

        return None

    def _check_ast_for_attribute(self, file_path: Path, attr_name: str):
        """
        Parses python file at file_path to check if attr_name is defined.
        """
        try:
            source = file_path.read_text("utf-8")
            tree = ast.parse(source, filename=str(file_path))

            found = False
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if node.name == attr_name:
                        found = True
                        break
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if target.id == attr_name:
                                found = True
                                break
            if not found:
                raise PackageValidationError(
                    f"Attribute '{attr_name}' not found in "
                    f"{file_path.name} (static analysis)."
                )
        except SyntaxError as e:
            raise PackageValidationError(
                f"Syntax error in package code {file_path}: {e}"
            )
        except Exception as e:
            raise PackageValidationError(f"Failed to parse entry point: {e}")

    @staticmethod
    def get_git_tag_version(package_dir: Path) -> str:
        """
        Gets the version from git tags in the package directory.

        Args:
            package_dir (Path): The directory containing the package.

        Returns:
            str: The version from the latest git tag.
        """
        try:
            importlib.import_module("git")
        except ImportError:
            logger.warning(
                "GitPython is required to get git tag version, "
                "using default version 0.0.1"
            )
            return "0.0.1"

        from git import Repo

        try:
            repo = Repo(package_dir)
            tags = repo.tags
            if tags:
                latest_tag = sorted(
                    tags, key=lambda t: t.commit.committed_datetime
                )[-1]
                return latest_tag.name
            logger.warning(
                f"No git tags found in {package_dir}, "
                "using default version 0.0.1"
            )
            return "0.0.1"
        except Exception as e:
            logger.warning(
                f"Failed to get git tag version from {package_dir}: {e}"
            )
            return "0.0.1"
