from __future__ import annotations
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
from dataclasses import asdict, dataclass, field
from .vectorization_config import TraceConfig

if TYPE_CHECKING:
    from ..importer.base_renderer import Renderer


@dataclass
class ImportSource:
    """
    A data record that links a WorkPiece back to its original source file
    and the configuration used to generate its vectors. This is not a DocItem.
    It distinguishes between the original file data and potentially modified
    working data (e.g., after a crop operation).
    """

    source_file: Path
    original_data: bytes
    renderer: "Renderer"
    working_data: Optional[bytes] = None
    vector_config: Optional[TraceConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if self.working_data is None:
            self.working_data = self.original_data

    @property
    def data(self) -> bytes:
        """The current working data for rendering and operations."""
        # This assertion satisfies the type checker and documents the
        # invariant that __post_init__ ensures working_data is never None.
        assert self.working_data is not None, (
            "working_data should have been initialized in __post_init__"
        )
        return self.working_data

    @data.setter
    def data(self, value: bytes):
        """Sets the current working data."""
        self.working_data = value

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the ImportSource to a dictionary."""
        return {
            "uid": self.uid,
            "source_file": str(self.source_file),
            "original_data": self.original_data,
            "working_data": self.working_data,
            "renderer_name": self.renderer.__class__.__name__,
            "vector_config": (
                asdict(self.vector_config) if self.vector_config else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> ImportSource:
        """Deserializes a dictionary into an ImportSource instance."""
        from ..importer import renderer_by_name

        config_data = state.get("vector_config")
        vector_config = TraceConfig(**config_data) if config_data else None
        renderer = renderer_by_name[state["renderer_name"]]

        # Handle backward compatibility for files saved before the
        # original_data/working_data split.
        if "data" in state and "original_data" not in state:
            original_data = state["data"]
        else:
            original_data = state["original_data"]

        # If working_data is missing from the dict, it will be passed as None
        # to the constructor, and __post_init__ will correctly set it.
        working_data = state.get("working_data")

        return cls(
            uid=state["uid"],
            source_file=Path(state["source_file"]),
            original_data=original_data,
            working_data=working_data,
            renderer=renderer,
            vector_config=vector_config,
            metadata=state.get("metadata", {}),
        )
