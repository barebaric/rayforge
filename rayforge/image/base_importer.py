from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..core.item import DocItem
    from ..core.vectorization_spec import VectorizationSpec
    from ..core.source_asset import SourceAsset
    from ..core.sketcher.sketch import Sketch


@dataclass
class ImportPayload:
    """
    A container for the complete result of a file import operation.
    It's a self-contained package ready for integration into a document.
    """

    source: "SourceAsset"
    items: List["DocItem"]
    sketches: List["Sketch"] = field(default_factory=list)


class Importer(ABC):
    """
    An abstract base class that defines the interface for all importers.

    An Importer acts as a factory, taking raw file data and producing a
    self-contained `ImportPayload`. This payload contains the `SourceAsset`
    (the link to the original file) and a list of `DocItem` objects
    (typically `WorkPiece` instances) ready to be added to a document.

    Architectural Contract:
    -----------------------
    To prevent "double transformation" bugs, all importers MUST follow a strict
    separation of concerns between an object's intrinsic shape and its
    physical transformation in the document.

    1.  **Generate Normalized Vectors**: The vector geometry created by the
        importer should represent the object's SHAPE, normalized to a standard
        unit size (e.g., fitting within a 1x1 box) while preserving the
        original aspect ratio.

    2.  **Assign to WorkPiece**: This normalized `Geometry` is assigned to
        `WorkPiece.boundaries`. At this point, the `WorkPiece`'s transformation
        matrix should be the identity matrix (scale=1).

    3.  **Apply Physical Size via Matrix**: The importer then determines the
        object's intended physical size in millimeters and calls
        `WorkPiece.set_size()`. This method correctly applies the physical
        dimensions by modifying the `WorkPiece.matrix`, scaling the
        normalized vectors to their final size.

    This ensures that the scale is applied only once, through the matrix,
    and that `WorkPiece.boundaries` remains a pure representation of shape.
    """

    label: str
    mime_types: Tuple[str, ...]
    extensions: Tuple[str, ...]
    is_bitmap: bool = False

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        """
        The constructor that all subclasses must implement.
        """
        self.raw_data = data
        self.source_file = source_file or Path("Untitled")

    @abstractmethod
    def get_doc_items(
        self, vectorization_spec: Optional["VectorizationSpec"] = None
    ) -> Optional[ImportPayload]:
        """
        Parses the raw data and returns a self-contained ImportPayload.

        The payload contains the single SourceAsset for the file and a list
        of top-level DocItems (WorkPieces and/or Groups). The generated
        WorkPieces should be linked to the SourceAsset via their
        `generation_config`.

        The returned items should be fully configured but unparented. Their
        transformation matrices should represent their position and scale
        within the document.

        If the importer cannot parse the data, it should return None.
        """
        raise NotImplementedError
