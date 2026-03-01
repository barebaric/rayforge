import logging
import re
import yaml
import semver
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union

from rayforge.core.hooks import PLUGIN_API_VERSION
from rayforge.shared.util.versioning import UnknownVersion, get_git_tag_version

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
        backend (Optional[str]): Module path loaded in worker and main
                                 processes (e.g., 'my_package.plugin').
        frontend (Optional[str]): Module path loaded only in main process
                                  (e.g., 'my_package.ui').
        assets (List[Dict[str, str]]): A list of asset definitions.
    """

    backend: Optional[str] = None
    frontend: Optional[str] = None
    assets: List[Dict[str, str]] = field(default_factory=list)


VersionType = Union[str, object]


@dataclass
class PackageMetadata:
    """
    Serializable metadata for a Rayforge package.
    """

    name: str
    description: str
    version: VersionType
    depends: List[str]
    author: PackageAuthor
    provides: PackageProvides
    api_version: int = 1
    url: str = ""
    display_name: str = ""
    requires: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts metadata back to a dictionary for YAML serialization."""
        result = asdict(self)
        if self.version is UnknownVersion:
            result["version"] = None
        return result

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

        requires = data.get("requires", [])
        if isinstance(requires, str):
            requires = [requires]

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
            api_version=data.get("api_version", PLUGIN_API_VERSION),
            requires=requires,
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
    def load_from_directory(
        cls,
        package_dir: Path,
        version: Optional[VersionType] = None,
    ) -> "Package":
        """
        Loads a package from a directory by parsing its YAML metadata file.
        The directory name is treated as the canonical package ID.

        Args:
            package_dir (Path): The directory containing the package.
            version (Optional[VersionType]): The version to use. If None,
                version is determined from git tags. For builtin packages
                that cannot determine version from git, pass UnknownVersion.

        Raises:
            RuntimeError: If version is None and no git tags are found.
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

            # 2. Parse 'Provides' (Backend, Frontend, and Assets)
            provides_data = data.get("provides", {})

            provides = PackageProvides(
                backend=provides_data.get("backend"),
                frontend=provides_data.get("frontend"),
                assets=provides_data.get("assets", []),
            )

            # 3. Construct Metadata
            # The directory name is the source of truth for the package
            # name (ID).
            depends = data.get("depends", [])
            if isinstance(depends, str):
                depends = [depends]

            requires = data.get("requires", [])
            if isinstance(requires, str):
                requires = [requires]

            resolved_version: VersionType
            if version is not None:
                resolved_version = version
            elif version is UnknownVersion:
                resolved_version = version
            else:
                resolved_version = get_git_tag_version(package_dir)

            metadata = PackageMetadata(
                name=package_dir.name,
                display_name=data.get("display_name", data.get("name", "")),
                description=data.get("description", ""),
                version=resolved_version,
                depends=depends,
                author=author,
                provides=provides,
                url=data.get("url", ""),
                api_version=data.get("api_version", PLUGIN_API_VERSION),
                requires=requires,
            )

            return cls(path=package_dir, metadata=metadata)

        except Exception as e:
            raise PackageValidationError(f"Structure error in metadata: {e}")

    def validate(self) -> bool:
        """
        Performs rigorous validation on the package.
        """
        logger.debug(f"Validating package structure for: {self.metadata.name}")

        # 1. Basic Field Validation
        if not self.metadata.name or not self.metadata.name.strip():
            raise PackageValidationError("Package 'name' cannot be empty.")

        if not self.metadata.description:
            logger.warning(
                f"Package '{self.metadata.name}' has no description."
            )

        # 2. Version Validation
        if self.metadata.version is UnknownVersion:
            pass
        else:
            try:
                version_str = str(self.metadata.version)
                clean_ver = version_str.lstrip("v")
                semver.VersionInfo.parse(clean_ver)
            except ValueError:
                raise PackageValidationError(
                    f"Invalid semantic version: {self.metadata.version}"
                )

        # 2.1 API Version Validation
        if not isinstance(self.metadata.api_version, int):
            raise PackageValidationError(
                f"api_version must be an integer, got: "
                f"{type(self.metadata.api_version).__name__}"
            )
        if self.metadata.api_version != PLUGIN_API_VERSION:
            raise PackageValidationError(
                f"Unsupported api_version: {self.metadata.api_version}. "
                f"Expected {PLUGIN_API_VERSION}."
            )

        # 2.2 Depends Validation
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

        # 3. Author Validation
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

        # 4. Asset Validation
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

        # 5. Backend Entry Point Validation
        if self.metadata.provides.backend:
            self._validate_entry_point(self.metadata.provides.backend)

        # 6. Frontend Entry Point Validation
        if self.metadata.provides.frontend:
            self._validate_entry_point(self.metadata.provides.frontend)

        return True

    def _validate_entry_point(self, entry_point: str):
        """
        Validates a module entry point.

        Entry point must be a valid Python module path
        like 'my_package.plugin'.
        """
        if not self._is_valid_module_path(entry_point):
            raise PackageValidationError(
                f"Entry point '{entry_point}' is not a valid module path. "
                "Use dotted notation (e.g., 'my_package.plugin')."
            )

        file_path = self._resolve_module_path(entry_point)
        if not file_path:
            raise PackageValidationError(
                f"Module '{entry_point}' not found in {self.root_path}"
            )

    def _is_valid_module_path(self, path: str) -> bool:
        """Check if a string is a valid Python module path."""
        if not path or path.startswith(".") or path.endswith("."):
            return False
        parts = path.split(".")
        return all(part.isidentifier() for part in parts)

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
