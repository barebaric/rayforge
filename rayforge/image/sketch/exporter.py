from __future__ import annotations
from typing import TYPE_CHECKING
from ..base_exporter import Exporter

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece


class SketchExporter(Exporter):
    """
    Exports the parametric source data of a sketch-based WorkPiece.
    """

    label = "Rayforge Sketch"
    extensions = (".rfs",)
    mime_types = ("application/vnd.rayforge-sketch",)

    def __init__(self, doc_item: "WorkPiece"):
        """
        Initializes the exporter for a specific sketch-based WorkPiece.

        Args:
            doc_item: The WorkPiece whose sketch source should be exported.
        """
        super().__init__(doc_item)
        # Type check to ensure we got a WorkPiece
        from ...core.workpiece import WorkPiece

        if not isinstance(doc_item, WorkPiece):
            raise TypeError("SketchExporter can only export WorkPiece items.")
        self.workpiece = doc_item

    def export(self) -> bytes:
        """
        Retrieves the serialized Sketch definition from the WorkPiece's
        SourceAsset.

        Returns:
            The raw JSON bytes representing the sketch.

        Raises:
            ValueError: If the WorkPiece is not derived from a sketch or if
                        the source data is missing.
        """
        source = self.workpiece.source
        if (
            not source
            or source.renderer.__class__.__name__ != "SketchImporter"
        ):
            raise ValueError(
                "Cannot export: The selected item is not based on a sketch."
            )

        if source.original_data is None:
            raise ValueError(
                "Cannot export: The sketch source data is missing."
            )

        return source.original_data
