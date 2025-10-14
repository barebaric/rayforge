"""Core material data structures for Rayforge."""

import logging
import yaml
import re
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MaterialAppearance:
    """Defines the visual properties of a material."""

    color: str = "#f0f0f0"
    pattern: str = "solid"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaterialAppearance":
        """Create an instance from a dictionary."""
        return cls(
            color=data.get("color", cls.color),
            pattern=data.get("pattern", "solid"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the appearance to a dictionary."""
        return {"color": self.color, "pattern": self.pattern}


@dataclass
class Material:
    """
    A pure data class representing a material in Rayforge.

    Materials define the visual and physical properties of stock items
    that can be cut or engraved.
    """

    uid: str
    name: str = ""
    description: str = ""
    category: str = ""
    appearance: MaterialAppearance = field(default_factory=MaterialAppearance)
    file_path: Optional[Path] = None

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.name:
            self.name = self.uid

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
            raise ValueError(
                f"Material file {file_path} must contain a dictionary"
            )

        # Extract required UID from filename or data
        uid = data.get("uid", file_path.stem)

        # Create material instance
        material = cls(
            uid=uid,
            name=data.get("name", uid),
            description=data.get("description", ""),
            category=data.get("category", ""),
            appearance=MaterialAppearance.from_dict(
                data.get("appearance", {})
            ),
            file_path=file_path,
        )

        return material

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the material to a dictionary representation.

        Returns:
            Dictionary containing all material data
        """
        return {
            "uid": self.uid,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "appearance": self.appearance.to_dict(),
        }

    def save_to_file(self, file_path: Optional[Path] = None) -> None:
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

    def __str__(self) -> str:
        """String representation of the material."""
        return f"Material(uid='{self.uid}', name='{self.name}')"

    def __repr__(self) -> str:
        """Detailed string representation of the material."""
        return (
            f"Material(uid='{self.uid}', name='{self.name}', "
            f"category='{self.category}', description='{self.description}')"
        )
