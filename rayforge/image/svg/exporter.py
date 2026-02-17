from __future__ import annotations
import math
from typing import List
from ...core.geo import Geometry
from ...core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    CMD_TYPE_BEZIER,
)
from ..base_exporter import GeometryExporter


class GeometrySvgExporter(GeometryExporter):
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
    ) -> str:
        path_data: List[str] = []
        _, _, _, max_y = geometry.rect()

        def transform(x: float, y: float) -> tuple:
            tx = x - min_x
            ty = max_y - y
            return tx, ty

        last_x = 0.0
        last_y = 0.0

        for cmd in geometry.iter_commands():
            cmd_type = cmd[0]
            x = cmd[1]
            y = cmd[2]

            if cmd_type == CMD_TYPE_MOVE:
                tx, ty = transform(x, y)
                path_data.append(f"M {tx:.6f} {ty:.6f}")
            elif cmd_type == CMD_TYPE_LINE:
                tx, ty = transform(x, y)
                path_data.append(f"L {tx:.6f} {ty:.6f}")
            elif cmd_type == CMD_TYPE_ARC:
                i = cmd[4]
                j = cmd[5]
                cw = bool(cmd[6])

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
            elif cmd_type == CMD_TYPE_BEZIER:
                c1x = cmd[4]
                c1y = cmd[5]
                c2x = cmd[6]
                c2y = cmd[7]

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
        cx = x1 + i
        cy = y1 + j

        start_angle = math.atan2(y1 - cy, x1 - cx)
        end_angle = math.atan2(y2 - cy, x2 - cx)

        if cw:
            arc_angle = start_angle - end_angle
        else:
            arc_angle = end_angle - start_angle

        while arc_angle < 0:
            arc_angle += 2 * math.pi
        while arc_angle > 2 * math.pi:
            arc_angle -= 2 * math.pi

        if arc_angle > math.pi:
            return 1
        return 0
