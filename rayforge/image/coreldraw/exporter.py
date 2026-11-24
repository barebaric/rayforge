from __future__ import annotations

from gettext import gettext as _

from raygeo.formats.coreldraw import write
from raygeo.geo import Geometry

from ..base_exporter import BaseExporter


class CorelDrawExporter(BaseExporter):
    """
    Exports a Geometry object to a CorelDRAW version 12 drawing.
    """

    label = _("CorelDRAW (CDR)")
    extensions = (".cdr",)
    mime_types = (
        "application/vnd.corel-draw",
        "image/x-coreldraw",
        "image/x-cdr",
        "application/x-cdr",
    )

    def __init__(self, geometry: Geometry):
        self.geometry = geometry

    def export(self) -> bytes:
        if self.geometry.is_empty():
            raise ValueError("Cannot export: The geometry is empty.")

        return write(self.geometry)


class MultiGeometryCorelDrawExporter(BaseExporter):
    """
    Exports multiple Geometry objects to a single CorelDRAW file.
    """

    label = _("CorelDRAW (CDR)")
    extensions = (".cdr",)
    mime_types = (
        "application/vnd.corel-draw",
        "image/x-coreldraw",
        "image/x-cdr",
        "application/x-cdr",
    )

    def __init__(self, geometries: list[Geometry]):
        self.geometries = geometries

    def export(self) -> bytes:
        non_empty = [g for g in self.geometries if not g.is_empty()]
        if not non_empty:
            raise ValueError("Cannot export: All geometries are empty.")

        merged = Geometry()
        for geometry in non_empty:
            merged.extend(geometry)
        return write(merged)
