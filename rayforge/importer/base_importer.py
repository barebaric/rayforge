from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.item import DocItem


class Importer(ABC):
    """
    An abstract base class that defines the interface for all importers.
    An Importer acts as a factory, taking raw file data and producing a
    list of DocItems (WorkPieces and/or Groups).
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
    def get_doc_items(self) -> "Optional[List[DocItem]]":
        """
        Parses the raw data and returns a list of top-level DocItems
        (WorkPieces and/or Groups).

        The returned items should be fully configured but unparented. Their
        transformation matrices should represent their position and scale
        within the document.

        If the importer cannot parse the data, it should return None or
        an empty list.
        """
        raise NotImplementedError
