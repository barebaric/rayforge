from __future__ import annotations
from typing import TYPE_CHECKING, cast
import json
from gettext import gettext as _
from rayforge.image.base_exporter import Exporter
from rayforge import const

if TYPE_CHECKING:
    from ..core import Sketch
    from rayforge.core.workpiece import WorkPiece


class SketchExporter(Exporter):
    """
    Exports the parametric source data of a sketch-based WorkPiece.
    """

    label = _("{app_name} Sketch").format(app_name=const.APP_NAME)
    extensions = (".rfs",)
    mime_types = (const.MIME_TYPE_SKETCH,)

    def __init__(self, doc_item: "WorkPiece"):
        """
        Initializes the exporter for a specific sketch-based WorkPiece.

        Args:
            doc_item: The WorkPiece whose sketch source should be exported.
        """
        super().__init__(doc_item)
        from rayforge.core.workpiece import WorkPiece

        if not isinstance(doc_item, WorkPiece):
            raise TypeError("SketchExporter can only export WorkPiece items.")
        self.workpiece = doc_item

    def export(self) -> bytes:
        """
        Retrieves the serialized Sketch definition from the document's
        sketch registry.

        Returns:
            The raw JSON bytes representing the sketch.

        Raises:
            ValueError: If the WorkPiece is not derived from a sketch or if
                        the sketch definition is missing.
        """
        sketch = cast("Sketch", self.workpiece.get_geometry_provider())
        if not sketch:
            raise ValueError(
                "Cannot export: The selected item is not based on a sketch "
                "or its definition is missing."
            )

        sketch_dict = sketch.to_dict()
        return json.dumps(sketch_dict).encode("utf-8")
