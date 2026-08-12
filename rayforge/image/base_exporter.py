from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.item import DocItem


class BaseExporter(ABC):
    """
    Abstract base class for exporters that work with Geometry objects.
    """

    label: str
    extensions: tuple[str, ...]
    mime_types: tuple[str, ...]

    @abstractmethod
    def export(self) -> bytes:
        """
        Performs the export operation.

        Returns:
            The exported data as a bytes object.
        """
        raise NotImplementedError


class Exporter(BaseExporter):
    """
    An abstract base class that defines the interface for all exporters.
    An exporter takes a DocItem and converts it to a specific file format
    represented as bytes.
    """

    label: str
    extensions: tuple[str, ...]
    mime_types: tuple[str, ...]

    def __init__(self, doc_item: "DocItem"):
        """
        Initializes the exporter with the document item to be exported.

        Args:
            doc_item: The DocItem instance to export.
        """
        self.doc_item = doc_item
