from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from ..core.item import DocItem
    from ..core.vectorization_config import TraceConfig
    from ..core.import_source import ImportSource


class ImportPayload(NamedTuple):
    """
    A container for the complete result of a file import operation.
    It's a self-contained package ready for integration into a document.
    """

    source: "ImportSource"
    items: List["DocItem"]


class Importer(ABC):
    """
    An abstract base class that defines the interface for all importers.
    An Importer acts as a factory, taking raw file data and producing a
    self-contained payload containing an ImportSource and a list of
    DocItems (WorkPieces and/or Groups).
    """

    label: str
    mime_types: Tuple[str, ...]
    extensions: Tuple[str, ...]

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        """
        The constructor that all subclasses must implement.
        """
        self.raw_data = data
        self.source_file = source_file or Path("Untitled")

    @abstractmethod
    def get_doc_items(
        self, vector_config: Optional["TraceConfig"] = None
    ) -> Optional[ImportPayload]:
        """
        Parses the raw data and returns a self-contained ImportPayload.

        The payload contains the single ImportSource for the file and a list
        of top-level DocItems (WorkPieces and/or Groups). The generated
        WorkPieces should be linked to the ImportSource via their
        `import_source_uid`.

        The returned items should be fully configured but unparented. Their
        transformation matrices should represent their position and scale
        within the document.

        If the importer cannot parse the data, it should return None.
        """
        raise NotImplementedError
