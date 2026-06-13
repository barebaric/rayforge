from __future__ import annotations

import io
import math
from gettext import gettext as _
from typing import List

import ezdxf
from raygeo.geo import Geometry
from raygeo.geo import Arc, Bezier, Line, Move
from raygeo.geo.shape.arc import get_arc_angles
from raygeo.geo.shape.bezier import linearize_bezier_segment

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

        for cmd in geometry.data:
            x = cmd.end[0]
            y = cmd.end[1]

            if isinstance(cmd, Move):
                flush_polyline()
            elif isinstance(cmd, Line):
                if not poly_points:
                    poly_points = [(last_x, last_y)]
                poly_points.append((x, y))
            elif isinstance(cmd, Arc):
                flush_polyline()

                i = cmd.center_offset[0]
                j = cmd.center_offset[1]
                cw = cmd.clockwise

                center_x = last_x + i
                center_y = last_y + j
                radius = math.hypot(i, j)
                if radius < 1e-9:
                    radius = 0.001

                start_angle, end_angle, _ = get_arc_angles(
                    (last_x, last_y), (x, y), (center_x, center_y), cw
                )
                start_angle = math.degrees(start_angle)
                end_angle = math.degrees(end_angle)

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
            elif isinstance(cmd, Bezier):
                flush_polyline()

                c1x = cmd.control1[0]
                c1y = cmd.control1[1]
                c2x = cmd.control2[0]
                c2y = cmd.control2[1]

                points = linearize_bezier_segment(
                    (last_x, last_y, 0.0),
                    (c1x, c1y, 0.0),
                    (c2x, c2y, 0.0),
                    (x, y, 0.0),
                    tolerance=0.1,
                )
                if points:
                    fit_points = [(p[0], p[1]) for p in points]
                    msp.add_spline_control_frame(
                        fit_points=fit_points, degree=3
                    )

            last_x = x
            last_y = y
        flush_polyline()


class MultiGeometryDxfExporter(BaseExporter):
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
