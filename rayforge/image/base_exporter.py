from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.item import DocItem


class Exporter(ABC):
    """
    An abstract base class that defines the interface for all exporters.
    An exporter takes a DocItem and converts it to a specific file format
    represented as bytes.
    """

    label: str
    extensions: Tuple[str, ...]
    mime_types: Tuple[str, ...]

    def __init__(self, doc_item: DocItem):
        """
        Initializes the exporter with the document item to be exported.

        Args:
            doc_item: The DocItem instance to export.
        """
        self.doc_item = doc_item

    @abstractmethod
    def export(self) -> bytes:
        """
        Performs the export operation.

        Returns:
            The exported data as a bytes object.
        """
        raise NotImplementedError
