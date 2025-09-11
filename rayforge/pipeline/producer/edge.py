from typing import Optional, TYPE_CHECKING, Tuple
import cairo
import numpy as np
import vtracer
from .base import OpsProducer
from ...core.ops import Ops
import xml.etree.ElementTree as ET
import re
import io

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece


def _get_binary_mask_from_alpha_channel(surface: cairo.ImageSurface) -> bytes:
    """
    Extracts the alpha channel from a surface, creates a high-contrast
    binary (black and white) mask from it, and returns the result as in-memory
    PNG bytes.
    """
    width = surface.get_width()
    height = surface.get_height()

    # Ensure the surface is in the expected format for direct buffer access
    if surface.get_format() != cairo.FORMAT_ARGB32:
        temp_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(temp_surface)
        ctx.set_source_surface(surface, 0, 0)
        ctx.paint()
        surface = temp_surface

    # Get a numpy view of the original pixel data
    data = surface.get_data()
    image_array = np.ndarray(
        shape=(height, width, 4), dtype=np.uint8, buffer=data
    )

    alpha_channel = image_array[:, :, 3].copy()

    # Use a 50% alpha threshold for a crisp, accurate boundary
    binary_mask_data = np.where(alpha_channel > 127, 0, 255).astype(np.uint8)

    # 1. Let Cairo create a surface with its own correctly-strided buffer.
    mask_surface = cairo.ImageSurface(cairo.FORMAT_A8, width, height)

    # 2. Get a NumPy view into Cairo's buffer. The shape must include the
    # stride.
    cairo_buffer = mask_surface.get_data()
    cairo_stride = mask_surface.get_stride()
    target_array = np.ndarray(
        shape=(height, cairo_stride), dtype=np.uint8, buffer=cairo_buffer
    )

    # 3. Copy our image data into the correctly-strided buffer, leaving any
    #    padding bytes at the end of each row untouched.
    target_array[:, :width] = binary_mask_data

    # Now mask_surface contains the correct data in the correct memory layout.

    # Write the mask surface to an in-memory PNG
    with io.BytesIO() as buffer:
        mask_surface.write_to_png(buffer)
        return buffer.getvalue()


class EdgeTracer(OpsProducer):
    """
    Uses the VTracer engine to trace all paths in a shape, including
    both external outlines and internal holes. This is the primary producer
    that interfaces with the vtracer library.
    """

    def run(
        self,
        laser,
        surface,
        pixels_per_mm,
        *,
        workpiece: "Optional[WorkPiece]" = None,
        y_offset_mm: float = 0.0,
    ) -> Ops:
        if workpiece and workpiece.source_ops:
            return workpiece.source_ops

        if (
            not surface
            or surface.get_width() == 0
            or surface.get_height() == 0
        ):
            return Ops()

        self.original_surface_height = surface.get_height()
        self.pixels_per_mm = pixels_per_mm

        # Create a binary mask directly from the alpha channel for accuracy.
        png_bytes = _get_binary_mask_from_alpha_channel(surface)

        # Use vtracer to convert the binary mask image to SVG paths.
        svg_str = vtracer.convert_raw_image_to_svg(
            img_bytes=png_bytes,
            img_format="png",
            colormode="binary",
            mode="polygon",
            filter_speckle=26,
            length_threshold=3.5,
        )

        return self._svg_to_ops(svg_str)

    def _parse_transform(self, transform_str: str) -> np.ndarray:
        matrix = np.identity(3)
        if not transform_str:
            return matrix
        match = re.search(
            r"translate\(\s*([-\d.eE]+)\s*,?\s*([-\d.eE]+)?\s*\)",
            transform_str,
        )
        if match:
            tx = float(match.group(1))
            ty = float(match.group(2)) if match.group(2) is not None else 0.0
            matrix[0, 2] = tx
            matrix[1, 2] = ty
        return matrix

    def _apply_transform(
        self, point: np.ndarray, matrix: np.ndarray
    ) -> np.ndarray:
        vec = np.array([point[0], point[1], 1])
        transformed_vec = matrix @ vec
        return transformed_vec[:2]

    def _svg_to_ops(self, svg_str: str) -> Ops:
        root = ET.fromstring(svg_str)
        final_ops = Ops()
        self._traverse_svg_node(root, np.identity(3), final_ops)
        return final_ops

    def _traverse_svg_node(
        self, node: ET.Element, parent_transform: np.ndarray, ops: Ops
    ):
        local_transform = self._parse_transform(node.get("transform", ""))
        transform = parent_transform @ local_transform

        tag = node.tag.split("}")[-1]
        if tag == "path":
            # The color filter is no longer needed, as `vtracer` now
            # provides the correct set of paths.
            path_data = node.get("d")
            if path_data:
                self._process_path(path_data, transform, ops)

        for child in node:
            self._traverse_svg_node(child, transform, ops)

    def _transform_point(self, p: Tuple[float, float]) -> Tuple[float, float]:
        px, py = p
        ops_py = self.original_surface_height - py
        if self.pixels_per_mm is None:
            return px, ops_py
        scale_x, scale_y = self.pixels_per_mm
        return (px / scale_x, ops_py / scale_y)

    def _process_path(
        self, path_data: str, transform: np.ndarray, ops: Ops
    ) -> None:
        tokens = re.findall(
            r"([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)", path_data
        )
        current_pos = np.array([0.0, 0.0])
        start_of_subpath = np.array([0.0, 0.0])

        for cmd, coords_str in tokens:
            coords = [
                float(c)
                for c in re.findall(
                    r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", coords_str
                )
            ]

            def add_transformed_line(p):
                tp = self._apply_transform(p, transform)
                ops.line_to(*self._transform_point(tuple(tp)))

            def add_transformed_move(p):
                tp = self._apply_transform(p, transform)
                ops.move_to(*self._transform_point(tuple(tp)))

            if cmd == "M":
                current_pos = np.array(coords[0:2])
                start_of_subpath = current_pos
                add_transformed_move(current_pos)
            elif cmd == "m":
                current_pos += np.array(coords[0:2])
                start_of_subpath = current_pos
                add_transformed_move(current_pos)
            elif cmd == "L":
                current_pos = np.array(coords[0:2])
                add_transformed_line(current_pos)
            elif cmd == "l":
                current_pos += np.array(coords[0:2])
                add_transformed_line(current_pos)
            elif cmd == "H":
                current_pos[0] = coords[0]
                add_transformed_line(current_pos)
            elif cmd == "h":
                current_pos[0] += coords[0]
                add_transformed_line(current_pos)
            elif cmd == "V":
                current_pos[1] = coords[0]
                add_transformed_line(current_pos)
            elif cmd == "v":
                current_pos[1] += coords[0]
                add_transformed_line(current_pos)
            elif cmd == "C":
                c1, c2, end = (
                    np.array(coords[0:2]),
                    np.array(coords[2:4]),
                    np.array(coords[4:6]),
                )
                self._flatten_bezier_segment(
                    current_pos, c1, c2, end, transform, ops
                )
                current_pos = end
            elif cmd == "c":
                c1, c2, end = (
                    current_pos + np.array(coords[0:2]),
                    current_pos + np.array(coords[2:4]),
                    current_pos + np.array(coords[4:6]),
                )
                self._flatten_bezier_segment(
                    current_pos, c1, c2, end, transform, ops
                )
                current_pos = end
            elif cmd.lower() == "z":
                if np.any(current_pos != start_of_subpath):
                    add_transformed_line(start_of_subpath)
                ops.close_path()
                current_pos = start_of_subpath

    def _flatten_bezier_segment(
        self, start, c1, c2, end, transform, ops, num_steps=20
    ):
        start_np, c1_np, c2_np, end_np = (
            np.array(start),
            np.array(c1),
            np.array(c2),
            np.array(end),
        )
        t_values = np.linspace(0, 1, num_steps)[1:]
        for t in t_values:
            one_minus_t = 1 - t
            p_px = (
                (one_minus_t**3 * start_np)
                + (3 * one_minus_t**2 * t * c1_np)
                + (3 * one_minus_t * t**2 * c2_np)
                + (t**3 * end_np)
            )
            tp = self._apply_transform(p_px, transform)
            ops.line_to(*self._transform_point(tuple(tp)))
