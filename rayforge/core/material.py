"""Core material data structures for Rayforge."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..shared.util.localized import LocalizedField

# Accept both plain strings and LocalizedField as input
LocalizedInput = str | LocalizedField

# Only WebP textures are supported
TEXTURE_EXTENSION = ".webp"

logger = logging.getLogger(__name__)


def _coerce_localized(value: LocalizedInput) -> LocalizedField:
    """
    Convert a plain string or dict into a LocalizedField.

    LocalizedField instances are passed through unchanged so their
    translations survive.
    """
    if isinstance(value, LocalizedField):
        return value
    return LocalizedField.from_yaml(value)


@dataclass
class MaterialAppearance:
    """Defines the visual properties of a material."""

    color: str = "#f0f0f0"
    pattern: str = "solid"
    texture: str | None = None
    texture_size_mm: float = 300.0
    roughness: float = 0.8
    metallic: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaterialAppearance":
        """Create an instance from a dictionary."""
        known_keys = {
            "color",
            "pattern",
            "texture",
            "texture_size_mm",
            "roughness",
            "metallic",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            color=data.get("color", cls.color),
            pattern=data.get("pattern", cls.pattern),
            texture=data.get("texture", cls.texture),
            texture_size_mm=data.get("texture_size_mm", cls.texture_size_mm),
            roughness=data.get("roughness", cls.roughness),
            metallic=data.get("metallic", cls.metallic),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the appearance to a dictionary."""
        result: dict[str, Any] = {"color": self.color, "pattern": self.pattern}
        if self.texture is not None:
            result["texture"] = self.texture
        result.update(
            {
                "texture_size_mm": self.texture_size_mm,
                "roughness": self.roughness,
                "metallic": self.metallic,
            }
        )
        result.update(self.extra)
        return result


@dataclass
class Material:
    """
    A pure data class representing a material in Rayforge.

    Materials define the visual and physical properties of stock items
    that can be cut or engraved.

    Note: LocalizedField handles all localization transparently.
    This class doesn't need to know about context or language.
    """

    uid: str
    name: LocalizedInput = field(default_factory=lambda: LocalizedField(""))
    description: LocalizedInput = field(
        default_factory=lambda: LocalizedField("")
    )
    category: LocalizedInput = field(
        default_factory=lambda: LocalizedField("")
    )
    appearance: MaterialAppearance = field(default_factory=MaterialAppearance)
    file_path: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Convert plain strings to LocalizedField
        if not isinstance(self.name, LocalizedField):
            self.name = LocalizedField(
                str(self.name) if self.name else self.uid
            )
        elif not self.name:
            self.name = LocalizedField(self.uid)

        if not isinstance(self.description, LocalizedField):
            self.description = LocalizedField(str(self.description))

        if not isinstance(self.category, LocalizedField):
            self.category = LocalizedField(str(self.category))

    @classmethod
    def from_file(cls, file_path: Path) -> "Material":
        """
        Create a Material instance from a YAML file.

        Args:
            file_path: Path to the YAML file containing material data

        Returns:
            Material instance with data loaded from the file

        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the file contains invalid YAML
            ValueError: If required fields are missing
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Material file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Invalid YAML in material file {file_path}: {e}"
            )

        if not isinstance(data, dict):
            raise TypeError(
                f"Material file {file_path} must contain a dictionary"
            )

        # Extract required UID from filename or data
        uid = data.get("uid", file_path.stem)

        # Extract known keys
        known_keys = {
            "uid",
            "name",
            "description",
            "category",
            "appearance",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        # Parse fields - LocalizedField handles both simple and localized
        # format
        name = LocalizedField.from_yaml(data.get("name", uid))
        description = LocalizedField.from_yaml(data.get("description", ""))
        category = LocalizedField.from_yaml(data.get("category", ""))

        material = cls(
            uid=uid,
            name=name,
            description=description,
            category=category,
            appearance=MaterialAppearance.from_dict(
                data.get("appearance", {})
            ),
            file_path=file_path,
            extra=extra,
        )

        return material

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the material to a dictionary representation.

        Returns:
            Dictionary containing all material data
        """
        result = {
            "uid": self.uid,
            "name": _coerce_localized(self.name).to_yaml(),
            "description": _coerce_localized(self.description).to_yaml(),
            "category": _coerce_localized(self.category).to_yaml(),
            "appearance": self.appearance.to_dict(),
        }
        result.update(self.extra)
        return result

    def save_to_file(self, file_path: Path | None = None) -> None:
        """
        Save the material to a YAML file.

        Args:
            file_path: Path to save the file. If None, uses self.file_path
        """
        target_path = file_path or self.file_path
        if not target_path:
            raise ValueError("No file path specified for saving material")

        # Ensure directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        with open(target_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        self.file_path = target_path
        logger.info(f"Saved material '{self.uid}' to {target_path}")

    def get_display_color(self) -> str:
        """
        Get the display color for the material.

        Returns:
            Hex color string or default if not specified
        """
        return self.appearance.color

    def get_display_rgba(
        self, alpha: float = 1.0
    ) -> tuple[float, float, float, float]:
        """
        Get the display color as RGBA tuple.

        Args:
            alpha: Alpha value (0.0 to 1.0)

        Returns:
            Tuple of (r, g, b, a) values in 0.0-1.0 range
        """
        color_hex = self.appearance.color
        color_pattern = r"^#?([a-fA-F0-9]{2})([a-fA-F0-9]{2})([a-fA-F0-9]{2})$"
        match = re.match(color_pattern, color_hex)
        if match:
            r, g, b = tuple(int(c, 16) / 255.0 for c in match.groups())
            return (r, g, b, alpha)
        else:
            # Fallback to default gray if color format is invalid
            return (0.5, 0.5, 0.5, alpha)

    def get_pattern(self) -> str:
        """
        Get the visual pattern for the material.

        Returns:
            Pattern name or 'solid' if not specified
        """
        return self.appearance.pattern

    def get_texture_path(self) -> Path | None:
        """
        Get the path to the material's WebP texture.

        Uses the appearance texture field when set, otherwise falls
        back to "<uid>.webp" next to the material's YAML file. Only
        WebP files are supported; the fallback is only returned when
        the file actually exists.

        Returns:
            Path to the texture file, or None if the material has no
            usable texture
        """
        if not self.file_path:
            return None
        directory = self.file_path.parent

        texture = self.appearance.texture
        if texture:
            if not self._is_safe_texture_name(texture):
                logger.warning(
                    "Ignoring unsupported texture '%s' for material "
                    "'%s' (only relative WebP paths are supported)",
                    texture,
                    self.uid,
                )
                return None
            return directory / texture

        candidate = directory / f"{self.uid}{TEXTURE_EXTENSION}"
        return candidate if candidate.is_file() else None

    @staticmethod
    def _is_safe_texture_name(name: str) -> bool:
        """Check that a texture name is a relative WebP path."""
        path = Path(name)
        return (
            not path.is_absolute()
            and path.suffix.lower() == TEXTURE_EXTENSION
            and ".." not in path.parts
        )

    def matches_search(self, query: str) -> bool:
        """
        Check if the material matches a search query in any language.

        Args:
            query: Search string (case-insensitive)

        Returns:
            True if query matches name, description, or category in any
            language
        """
        return (
            _coerce_localized(self.name).matches(query)
            or _coerce_localized(self.description).matches(query)
            or _coerce_localized(self.category).matches(query)
        )

    def __str__(self) -> str:
        """String representation of the material."""
        return f"Material(uid='{self.uid}', name='{self.name}')"

    def __repr__(self) -> str:
        """Detailed string representation of the material."""
        return (
            f"Material(uid='{self.uid}', name='{self.name}', "
            f"category='{self.category}', description='{self.description}')"
        )
