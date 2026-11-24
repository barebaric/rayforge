from __future__ import annotations

from gettext import gettext as _

from raygeo.formats.dxf.writer import write
from raygeo.geo import Geometry

from ..base_exporter import BaseExporter


class GeometryDxfExporter(BaseExporter):
    """
    Exports a Geometry object to DXF format.
    """

    label = _("DXF (CAD Exchange Format)")
    extensions = (".dxf",)
    mime_types = ("image/vnd.dxf",)

    def __init__(self, geometry: Geometry):
        self.geometry = geometry

    def export(self) -> bytes:
        if self.geometry.is_empty():
            raise ValueError("Cannot export: The geometry is empty.")

        return write(self.geometry).encode("utf-8")


class MultiGeometryDxfExporter(BaseExporter):
    """
    Exports multiple Geometry objects to a single DXF file.
    """

    label = _("DXF (CAD Exchange Format)")
    extensions = (".dxf",)
    mime_types = ("image/vnd.dxf",)

    def __init__(self, geometries: list[Geometry]):
        self.geometries = geometries

    def export(self) -> bytes:
        non_empty = [g for g in self.geometries if not g.is_empty()]
        if not non_empty:
            raise ValueError("Cannot export: All geometries are empty.")

        merged = Geometry()
        for geometry in non_empty:
            merged.extend(geometry)
        return write(merged).encode("utf-8")
