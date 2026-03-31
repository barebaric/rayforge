"""Core 3D model data structures for Rayforge."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelLibrary:
    """
    Represents a source of 3D models.

    Each library has a root ``path`` on the filesystem.  Models are
    files within the category subdirectories of that path.

    Attributes:
        library_id: Unique identifier (e.g. "user", "core").
        display_name: Human-readable name shown in the UI.
        path: Root directory containing category subdirectories.
        read_only: Whether models can be added/removed.
        icon_name: Icon name for the UI row.
    """

    library_id: str
    display_name: str
    path: Path
    read_only: bool = False
    icon_name: str = "folder-symbolic"


@dataclass(frozen=True)
class ModelCategory:
    """
    A fixed model category (e.g. machines, chucks).

    Categories are defined once by the application and are the same
    across all libraries.  Each category maps to a subdirectory name
    inside every library root.

    Attributes:
        id: Directory name used as subdirectory (e.g. ``"machines"``).
        display_name: Internationalized human-readable name.
        description: Internationalized short description.
    """

    id: str
    display_name: str
    description: str = ""

    def __str__(self) -> str:
        return self.display_name


@dataclass
class Model:
    """
    A data class representing a 3D model asset in Rayforge.

    Models are referenced by a relative path (e.g.
    ``Path("chucks/abc.glb")``) and resolved by the
    ``ModelManager`` by searching registered libraries in order.
    """

    name: str
    path: Path
    description: str = ""
    file_path: Optional[Path] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Model":
        """
        Create a Model instance from a dictionary.

        Args:
            data: Dictionary with model data. Must contain at
                least ``name`` and ``path``.

        Returns:
            A new Model instance.
        """
        known_keys = {"name", "path", "description"}
        extra = {k: v for k, v in data.items() if k not in known_keys}

        raw_path = data.get("path", "")
        return cls(
            name=data.get("name", ""),
            path=Path(raw_path) if raw_path else Path(),
            description=data.get("description", ""),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary representation.

        Returns:
            Dictionary containing all model data.
        """
        result: Dict[str, Any] = {
            "name": self.name,
            "path": str(self.path),
            "description": self.description,
        }
        result.update(self.extra)
        return result

    def __str__(self) -> str:
        return f"Model(name='{self.name}', path='{self.path}')"
