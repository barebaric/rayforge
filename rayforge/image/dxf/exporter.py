from __future__ import annotations
import io
import math
from typing import List
import ezdxf
from ...core.geo import Geometry
from ...core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    CMD_TYPE_BEZIER,
)
from ..base_exporter import GeometryExporter


class GeometryDxfExporter(GeometryExporter):
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

        return self._geometry_to_dxf(self.geometry)

    def _geometry_to_dxf(self, geometry: Geometry) -> bytes:
        doc = ezdxf.new()  # type: ignore[attr-defined]
        doc.header["$INSUNITS"] = 4  # Millimeters
        msp = doc.modelspace()
        self._add_geometry_to_msp(geometry, msp)

        output = io.StringIO()
        doc.write(output)
        return output.getvalue().encode("utf-8")

    def _add_geometry_to_msp(self, geometry: Geometry, msp) -> None:
        """Add geometry entities to a DXF modelspace."""
        last_x = 0.0
        last_y = 0.0
        poly_points: List[tuple] = []

        def flush_polyline():
            nonlocal poly_points
            if len(poly_points) >= 2:
                msp.add_lwpolyline(poly_points)
            poly_points = []

        for cmd in geometry.iter_commands():
            cmd_type = cmd[0]
            x = cmd[1]
            y = cmd[2]

            if cmd_type == CMD_TYPE_MOVE:
                flush_polyline()
            elif cmd_type == CMD_TYPE_LINE:
                if not poly_points:
                    poly_points = [(last_x, last_y)]
                poly_points.append((x, y))
            elif cmd_type == CMD_TYPE_ARC:
                flush_polyline()

                i = cmd[4]
                j = cmd[5]
                cw = cmd[6]

                center_x = last_x + i
                center_y = last_y + j
                radius = math.hypot(i, j)
                if radius < 1e-9:
                    radius = 0.001

                start_angle = math.degrees(
                    math.atan2(last_y - center_y, last_x - center_x)
                )
                end_angle = math.degrees(
                    math.atan2(y - center_y, x - center_x)
                )

                if cw:
                    start_angle, end_angle = end_angle, start_angle
                if end_angle <= start_angle:
                    end_angle += 360

                msp.add_arc(
                    center=(center_x, center_y),
                    radius=radius,
                    start_angle=start_angle,
                    end_angle=end_angle,
                )
            elif cmd_type == CMD_TYPE_BEZIER:
                flush_polyline()

                c1x = cmd[4]
                c1y = cmd[5]
                c2x = cmd[6]
                c2y = cmd[7]

                points = self._bezier_to_points(
                    last_x, last_y, c1x, c1y, c2x, c2y, x, y
                )
                if points:
                    fit_points = [(p[0], p[1]) for p in points]
                    msp.add_spline_control_frame(
                        fit_points=fit_points, degree=3
                    )

            last_x = x
            last_y = y

        flush_polyline()

    def _bezier_to_points(
        self,
        x0: float,
        y0: float,
        c1x: float,
        c1y: float,
        c2x: float,
        c2y: float,
        x1: float,
        y1: float,
        segments: int = 20,
    ) -> List[tuple]:
        points = [(x0, y0)]
        for i in range(1, segments):
            t = i / segments
            t2 = t * t
            t3 = t2 * t
            mt = 1 - t
            mt2 = mt * mt
            mt3 = mt2 * mt

            px = mt3 * x0 + 3 * mt2 * t * c1x + 3 * mt * t2 * c2x + t3 * x1
            py = mt3 * y0 + 3 * mt2 * t * c1y + 3 * mt * t2 * c2y + t3 * y1
            points.append((px, py))
        points.append((x1, y1))
        return points


class MultiGeometryDxfExporter(GeometryExporter):
    """
    Exports multiple Geometry objects to a single DXF file.
    """

    label = _("DXF (CAD Exchange Format)")
    extensions = (".dxf",)
    mime_types = ("image/vnd.dxf",)

    def __init__(self, geometries: List[Geometry]):
        self.geometries = geometries

    def export(self) -> bytes:
        non_empty = [g for g in self.geometries if not g.is_empty()]
        if not non_empty:
            raise ValueError("Cannot export: All geometries are empty.")

        doc = ezdxf.new()  # type: ignore[attr-defined]
        doc.header["$INSUNITS"] = 4  # Millimeters
        msp = doc.modelspace()

        single_exporter = GeometryDxfExporter(Geometry())
        for geo in non_empty:
            single_exporter._add_geometry_to_msp(geo, msp)

        output = io.StringIO()
        doc.write(output)
        return output.getvalue().encode("utf-8")
