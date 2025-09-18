from __future__ import annotations
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict
from .vectorization_config import TraceConfig


class ImportSource:
    """
    A data record that links a WorkPiece back to its original source file
    and the configuration used to generate its vectors. This is not a DocItem.
    """

    def __init__(
        self,
        source_file: Path,
        data: bytes,
        vector_config: Optional[TraceConfig] = None,
        uid: Optional[str] = None,
    ):
        self.uid: str = uid or str(uuid.uuid4())
        self.source_file = source_file
        self.data = data
        self.vector_config = vector_config

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the ImportSource to a dictionary."""
        return {
            "uid": self.uid,
            "source_file": str(self.source_file),
            "data": self.data,
            "vector_config": (
                asdict(self.vector_config) if self.vector_config else None
            ),
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> ImportSource:
        """Deserializes a dictionary into an ImportSource instance."""
        config_data = state.get("vector_config")
        vector_config = TraceConfig(**config_data) if config_data else None

        return cls(
            uid=state["uid"],
            source_file=Path(state["source_file"]),
            data=state["data"],
            vector_config=vector_config,
        )
