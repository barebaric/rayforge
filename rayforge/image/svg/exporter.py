from __future__ import annotations

import math
from gettext import gettext as _
from typing import List

from raygeo import Geometry
from raygeo.geo import Arc, Bezier, Line, Move
from raygeo.geo.shape.arc import get_arc_angles
from raygeo.geo.shape.rect import get_combined_rect

from ..base_exporter import BaseExporter


class GeometrySvgExporter(BaseExporter):
    """
    Exports a Geometry object to SVG format.
    """

    label = _("SVG (Scalable Vector Graphics)")
    extensions = (".svg",)
    mime_types = ("image/svg+xml",)

    def __init__(self, geometry: Geometry):
        self.geometry = geometry

    def export(self) -> bytes:
        if self.geometry.is_empty():
            raise ValueError("Cannot export: The geometry is empty.")

        min_x, min_y, max_x, max_y = self.geometry.rect()
        width = max(max_x - min_x, 1e-9)
        height = max(max_y - min_y, 1e-9)

        svg_content = self._geometry_to_svg(
            self.geometry, min_x, min_y, width, height
        )
        return svg_content.encode("utf-8")

    def _geometry_to_svg(
        self,
        geometry: Geometry,
        min_x: float,
        min_y: float,
        width: float,
        height: float,
    ) -> str:
        path_data = self._geometry_to_svg_path(geometry, min_x, min_y)

        padding = 1.0
        svg_width = width + 2 * padding
        svg_height = height + 2 * padding

        vb = (
            f'viewBox="{-padding} {-padding} {svg_width:.3f} {svg_height:.3f}"'
        )
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{svg_width:.3f}mm" height="{svg_height:.3f}mm" {vb}>'
        ]

        if path_data:
            svg_parts.append(
                f'<path d="{path_data}" fill="none" stroke="black" '
                f'stroke-width="0.1" stroke-linecap="round" '
                f'stroke-linejoin="round" />'
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def _geometry_to_svg_path(
        self,
        geometry: Geometry,
        min_x: float,
        min_y: float,
        max_y: float | None = None,
    ) -> str:
        path_data: List[str] = []
        if max_y is None:
            _, _, _, max_y = geometry.rect()

        def transform(x: float, y: float) -> tuple:
            tx = x - min_x
            assert max_y is not None
            ty = max_y - y
            return tx, ty

        last_x = 0.0
        last_y = 0.0

        for cmd in geometry.data:
            x = cmd.end[0]
            y = cmd.end[1]

            if isinstance(cmd, Move):
                tx, ty = transform(x, y)
                path_data.append(f"M {tx:.6f} {ty:.6f}")
            elif isinstance(cmd, Line):
                tx, ty = transform(x, y)
                path_data.append(f"L {tx:.6f} {ty:.6f}")
            elif isinstance(cmd, Arc):
                i = cmd.center_offset[0]
                j = cmd.center_offset[1]
                cw = cmd.clockwise

                radius = math.hypot(i, j)
                large_arc = self._compute_large_arc_flag(
                    last_x, last_y, x, y, i, j, cw
                )
                sweep = 1 if cw else 0

                tx, ty = transform(x, y)
                path_data.append(
                    f"A {radius:.6f} {radius:.6f} 0 {large_arc} {sweep} "
                    f"{tx:.6f} {ty:.6f}"
                )
            elif isinstance(cmd, Bezier):
                c1x = cmd.control1[0]
                c1y = cmd.control1[1]
                c2x = cmd.control2[0]
                c2y = cmd.control2[1]

                tx, ty = transform(x, y)
                c1tx, c1ty = transform(c1x, c1y)
                c2tx, c2ty = transform(c2x, c2y)
                path_data.append(
                    f"C {c1tx:.6f} {c1ty:.6f} "
                    f"{c2tx:.6f} {c2ty:.6f} "
                    f"{tx:.6f} {ty:.6f}"
                )

            last_x = x
            last_y = y

        return " ".join(path_data)

    def _compute_large_arc_flag(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        i: float,
        j: float,
        cw: bool,
    ) -> int:
        _, _, sweep = get_arc_angles((x1, y1), (x2, y2), (x1 + i, y1 + j), cw)
        return 1 if abs(sweep) > math.pi else 0


class MultiGeometrySvgExporter(BaseExporter):
    """
    Exports multiple Geometry objects to a single SVG file.
    """

    label = _("SVG (Scalable Vector Graphics)")
    extensions = (".svg",)
    mime_types = ("image/svg+xml",)

    def __init__(self, geometries: List[Geometry]):
        self.geometries = geometries

    def export(self) -> bytes:
        non_empty = [g for g in self.geometries if not g.is_empty()]
        if not non_empty:
            raise ValueError("Cannot export: All geometries are empty.")

        min_x, min_y, max_x, max_y = get_combined_rect(non_empty)

        width = max(max_x - min_x, 1e-9)
        height = max(max_y - min_y, 1e-9)

        svg_content = self._geometries_to_svg(
            non_empty, min_x, min_y, width, height
        )
        return svg_content.encode("utf-8")

    def _geometries_to_svg(
        self,
        geometries: List[Geometry],
        min_x: float,
        min_y: float,
        width: float,
        height: float,
    ) -> str:
        padding = 1.0
        svg_width = width + 2 * padding
        svg_height = height + 2 * padding
        max_y = min_y + height

        vb = (
            f'viewBox="{-padding} {-padding} {svg_width:.3f} {svg_height:.3f}"'
        )
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{svg_width:.3f}mm" height="{svg_height:.3f}mm" {vb}>'
        ]

        single_exporter = GeometrySvgExporter(Geometry())
        for geo in geometries:
            path_data = single_exporter._geometry_to_svg_path(
                geo, min_x, min_y, max_y
            )
            if path_data:
                svg_parts.append(
                    f'<path d="{path_data}" fill="none" stroke="black" '
                    f'stroke-width="0.1" stroke-linecap="round" '
                    f'stroke-linejoin="round" />'
                )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)
