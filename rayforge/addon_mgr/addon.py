import logging
import re
import yaml
import semver
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union

from rayforge.core.hooks import MINIMUM_API_VERSION, PLUGIN_API_VERSION
from rayforge.shared.util.versioning import UnknownVersion, get_git_tag_version

logger = logging.getLogger(__name__)

METADATA_FILENAME = "rayforge-addon.yaml"


class AddonValidationError(Exception):
    """
    Raised when an addon fails validation checks.
    """

    pass


@dataclass
class AddonAuthor:
    """Represents the author of an addon."""

    name: str
    email: str


@dataclass
class AddonLicense:
    """
    Represents licensing information for an addon.

    Attributes:
        name: SPDX license identifier (e.g., "MIT", "BSL-1.1").
        required: Whether a license is required for use.
        purchase_url: URL to purchase a license.
        product_id: Product identifier for license validation.
        product_ids: List of product identifiers for license validation.
        patreon_tier_ids: List of Patreon tier IDs that grant access.
    """

    name: str = ""
    required: bool = False
    purchase_url: str = ""
    product_id: str = ""
    product_ids: List[str] = field(default_factory=list)
    patreon_tier_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls, data: Optional[Dict[str, Any]]
    ) -> Optional["AddonLicense"]:
        """Creates an AddonLicense from a dictionary, or None if empty."""
        if not data:
            return None
        if not isinstance(data, dict):
            return None
        tier_ids = data.get("patreon_tier_ids", [])
        if tier_ids is None:
            tier_ids = []
        product_ids = data.get("product_ids", [])
        if product_ids is None:
            product_ids = []
        return cls(
            name=data.get("name", ""),
            required=bool(data.get("required", False)),
            purchase_url=data.get("purchase_url", ""),
            product_id=data.get("product_id", ""),
            product_ids=product_ids if isinstance(product_ids, list) else [],
            patreon_tier_ids=tier_ids if isinstance(tier_ids, list) else [],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the license to a dictionary for serialization."""
        result = {}
        if self.name:
            result["name"] = self.name
        if self.required:
            result["required"] = self.required
        if self.purchase_url:
            result["purchase_url"] = self.purchase_url
        if self.product_id:
            result["product_id"] = self.product_id
        if self.product_ids:
            result["product_ids"] = self.product_ids
        if self.patreon_tier_ids:
            result["patreon_tier_ids"] = self.patreon_tier_ids
        return result

    def get_all_product_ids(self) -> List[str]:
        """Returns all product IDs (both single and list)."""
        ids = list(self.product_ids)
        if self.product_id:
            ids.append(self.product_id)
        return ids


@dataclass
class AddonProvides:
    """
    Defines what the addon provides to the system.

    Attributes:
        backend (Optional[str]): Module path loaded in worker and main
                                 processes (e.g., 'my_addon.plugin').
        frontend (Optional[str]): Module path loaded only in main process
                                  (e.g., 'my_addon.ui').
        assets (List[Dict[str, str]]): A list of asset definitions.
    """

    backend: Optional[str] = None
    frontend: Optional[str] = None
    assets: List[Dict[str, str]] = field(default_factory=list)


VersionType = Union[str, object]


@dataclass
class AddonMetadata:
    """
    Serializable metadata for a Rayforge addon.
    """

    name: str
    description: str
    version: VersionType
    depends: List[str]
    author: AddonAuthor
    provides: AddonProvides
    api_version: int = 1
    url: str = ""
    display_name: str = ""
    requires: List[str] = field(default_factory=list)
    license: Optional[AddonLicense] = None

    @property
    def license_name(self) -> str:
        """Returns the license name if available."""
        return self.license.name if self.license else ""

    def to_dict(self) -> Dict[str, Any]:
        """Converts metadata back to a dictionary for YAML serialization."""
        result = asdict(self)
        if self.version is UnknownVersion:
            result["version"] = None
        if self.license:
            result["license"] = self.license.to_dict()
        return result

    @classmethod
    def from_registry_entry(
        cls, addon_name: str, data: Dict[str, Any]
    ) -> "AddonMetadata":
        """
        Parses a registry dictionary entry into an AddonMetadata object.
        Handles normalization of author fields and mapping keys.
        """
        author_info = data.get("author", {})
        if isinstance(author_info, dict):
            author = AddonAuthor(
                name=author_info.get("name", ""),
                email=author_info.get("email", ""),
            )
        elif isinstance(author_info, str):
            match = re.match(r"(.*) <(.*)>", author_info)
            if match:
                author = AddonAuthor(
                    name=match.group(1).strip(), email=match.group(2).strip()
                )
            else:
                author = AddonAuthor(name=author_info, email="")
        else:
            author = AddonAuthor(name="", email="")

        provides = AddonProvides()

        depends = data.get("depends", [])
        if isinstance(depends, str):
            depends = [depends]

        requires = data.get("requires", [])
        if isinstance(requires, str):
            requires = [requires]

        return cls(
            name=addon_name,
            display_name=data.get("display_name", addon_name),
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
            license=AddonLicense.from_dict(data.get("license")),
        )


class Addon:
    """
    A class representing a loadable Rayforge addon.
    """

    def __init__(self, path: Path, metadata: AddonMetadata):
        """
        Initialize the Addon.

        Args:
            path (Path): The root directory of the addon on the filesystem.
            metadata (AddonMetadata): The parsed metadata object.
        """
        self.root_path = path
        self.metadata = metadata
        self.license_message: str = ""
        self.purchase_url: str = ""

    @classmethod
    def load_from_directory(
        cls,
        addon_dir: Path,
        version: Optional[VersionType] = None,
    ) -> "Addon":
        """
        Loads an addon from a directory by parsing its YAML metadata file.

        Args:
            addon_dir (Path): The directory containing the addon.
            version (Optional[VersionType]): The version to use. If None,
                version is determined from git tags. For builtin addons
                that cannot determine version from git, pass UnknownVersion.

        Raises:
            FileNotFoundError: If the metadata file is missing.
            AddonValidationError: If parsing fails or required fields
              are missing.
            RuntimeError: If version is None and no git tags are found.
        """
        meta_file = addon_dir / METADATA_FILENAME

        if not meta_file.exists():
            raise FileNotFoundError(
                f"No addon metadata file ('{METADATA_FILENAME}') "
                f"found in {addon_dir}"
            )

        try:
            with open(meta_file, "r") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise AddonValidationError(f"Failed to parse YAML metadata: {e}")

        try:
            # The addon ID (namespace) MUST be defined in the metadata.
            addon_name = data.get("name")
            if not addon_name or not str(addon_name).strip():
                raise AddonValidationError(
                    f"Metadata file '{METADATA_FILENAME}' in {addon_dir} "
                    "is missing the required 'name' field."
                )
            addon_name = str(addon_name).strip()

            author_data = data.get("author", {})
            if isinstance(author_data, str):
                match = re.match(r"(.*) <(.*)>", author_data)
                if match:
                    author = AddonAuthor(
                        name=match.group(1).strip(),
                        email=match.group(2).strip(),
                    )
                else:
                    author = AddonAuthor(name=author_data, email="")
            else:
                author = AddonAuthor(
                    name=author_data.get("name", ""),
                    email=author_data.get("email", ""),
                )

            provides_data = data.get("provides", {})

            provides = AddonProvides(
                backend=provides_data.get("backend"),
                frontend=provides_data.get("frontend"),
                assets=provides_data.get("assets", []),
            )

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
                resolved_version = get_git_tag_version(addon_dir)

            metadata = AddonMetadata(
                name=addon_name,
                display_name=data.get("display_name", addon_name),
                description=data.get("description", ""),
                version=resolved_version,
                depends=depends,
                author=author,
                provides=provides,
                url=data.get("url", ""),
                api_version=data.get("api_version", PLUGIN_API_VERSION),
                requires=requires,
                license=AddonLicense.from_dict(data.get("license")),
            )

            return cls(path=addon_dir, metadata=metadata)

        except AddonValidationError:
            raise
        except Exception as e:
            raise AddonValidationError(f"Structure error in metadata: {e}")

    def validate(self) -> bool:
        """
        Performs rigorous validation on the addon.
        """
        logger.debug(f"Validating addon structure for: {self.metadata.name}")

        if not self.metadata.name or not self.metadata.name.strip():
            raise AddonValidationError("Addon 'name' cannot be empty.")

        if not self.metadata.name.isidentifier():
            raise AddonValidationError(
                f"Addon name '{self.metadata.name}' is not a valid Python "
                "identifier. It must contain only letters, numbers, and "
                "underscores, and cannot start with a number."
            )

        if not self.metadata.description:
            logger.warning(f"Addon '{self.metadata.name}' has no description.")

        if self.metadata.version is UnknownVersion:
            pass
        else:
            try:
                version_str = str(self.metadata.version)
                clean_ver = version_str.lstrip("v")
                semver.VersionInfo.parse(clean_ver)
            except ValueError:
                raise AddonValidationError(
                    f"Invalid semantic version: {self.metadata.version}"
                )

        if not isinstance(self.metadata.api_version, int):
            raise AddonValidationError(
                f"api_version must be an integer, got: "
                f"{type(self.metadata.api_version).__name__}"
            )
        if self.metadata.api_version < MINIMUM_API_VERSION:
            raise AddonValidationError(
                f"Unsupported api_version: {self.metadata.api_version}. "
                f"Minimum supported version is {MINIMUM_API_VERSION}."
            )
        if self.metadata.api_version > PLUGIN_API_VERSION:
            raise AddonValidationError(
                f"Unsupported api_version: {self.metadata.api_version}. "
                f"Maximum supported version is {PLUGIN_API_VERSION}."
            )

        for dep in self.metadata.depends:
            if not isinstance(dep, str):
                raise AddonValidationError(
                    f"Dependency must be a string: {dep}"
                )
            parts = dep.split(",")
            if not parts or not parts[0]:
                raise AddonValidationError(f"Invalid dependency format: {dep}")
            pkg_part = parts[0].strip()
            if not pkg_part:
                raise AddonValidationError(f"Invalid dependency format: {dep}")
            for constraint in parts[1:]:
                constraint = constraint.strip()
                if not constraint:
                    continue
                op_match = re.match(r"^([~^><=!]+)(.+)$", constraint)
                if not op_match:
                    raise AddonValidationError(
                        f"Invalid version constraint '{constraint}' in: {dep}"
                    )
                version_str = op_match.group(2).lstrip("v")
                operator = op_match.group(1)

                if operator == "~":
                    version_parts = version_str.split(".")
                    if len(version_parts) == 2:
                        version_str = f"{version_str}.0"
                    elif len(version_parts) == 1:
                        version_str = f"{version_str}.0.0"

                try:
                    semver.VersionInfo.parse(version_str)
                except ValueError:
                    raise AddonValidationError(
                        f"Invalid semantic version in constraint "
                        f"'{constraint}': {dep}"
                    )

        if not self.metadata.author.name:
            raise AddonValidationError("Author name is required.")

        if "your-github-username" in self.metadata.author.name.lower():
            raise AddonValidationError("Placeholder detected in author name.")

        if self.metadata.author.email:
            if not re.match(
                r"^[^@\s]+@[^@\s]+\.[^@\s]+$", self.metadata.author.email
            ):
                logger.warning(
                    f"Author email '{self.metadata.author.email}' "
                    "appears invalid."
                )

        for asset in self.metadata.provides.assets:
            path_str = asset.get("path")
            if not path_str:
                raise AddonValidationError(
                    "Asset entry is missing 'path' key."
                )

            if ".." in path_str or path_str.startswith("/"):
                raise AddonValidationError(
                    f"Invalid asset path '{path_str}'. Paths must be relative."
                )

            full_path = self.root_path / path_str
            if not full_path.exists():
                raise AddonValidationError(f"Asset path not found: {path_str}")

        if self.metadata.provides.backend:
            self._validate_entry_point(self.metadata.provides.backend)

        if self.metadata.provides.frontend:
            self._validate_entry_point(self.metadata.provides.frontend)

        return True

    def _validate_entry_point(self, entry_point: str):
        """
        Validates a module entry point.

        Entry point must be a valid Python module path
        like 'my_addon.plugin'.
        """
        if not self._is_valid_module_path(entry_point):
            raise AddonValidationError(
                f"Entry point '{entry_point}' is not a valid module path. "
                "Use dotted notation (e.g., 'my_addon.plugin')."
            )

        file_path = self._resolve_module_path(entry_point)
        if not file_path:
            raise AddonValidationError(
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
        direct_file_path = self.root_path / module_str
        if direct_file_path.exists():
            return direct_file_path

        rel_path = module_str.replace(".", "/")

        path_init = self.root_path / rel_path / "__init__.py"
        if path_init.exists():
            return path_init

        path_py = self.root_path / (rel_path + ".py")
        if path_py.exists():
            return path_py

        return None
