from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from ..base_importer import Importer, ImportPayload
from ...core.sketcher import Sketch

if TYPE_CHECKING:
    from ...core.vectorization_spec import VectorizationSpec


class SketchImporter(Importer):
    """
    Parses a .rfs file (serialized Sketch data) and prepares it for
    integration into a document.
    """

    label = "Rayforge Sketch"
    extensions = (".rfs",)
    mime_types = ("application/vnd.rayforge-sketch",)
    is_bitmap = False

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)
        # Temporary property for testing as per Step 2 of the plan.
        self.parsed_sketch: Optional[Sketch] = None

    def get_doc_items(
        self, vectorization_spec: Optional["VectorizationSpec"] = None
    ) -> Optional[ImportPayload]:
        """
        Deserializes the raw sketch data.

        For Step 2 of the implementation plan, this method only parses the
        sketch and stores it for verification. It does not yet generate
        the full ImportPayload with a WorkPiece.
        """
        try:
            sketch_dict = json.loads(self.raw_data.decode("utf-8"))
            self.parsed_sketch = Sketch.from_dict(sketch_dict)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Failed to parse, not a valid sketch file.
            return None

        # Per Step 2, return None for now. In a future step, this will
        # solve the sketch and return a full ImportPayload.
        return None
