import cairo
import logging
from typing import Optional, Tuple
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.ops import Ops
from .parser import RuidaParser
from .job import RuidaJob
from ..renderer import Renderer, CAIRO_MAX_DIMENSION

logger = logging.getLogger(__name__)


class RuidaRenderer(Renderer):
    label = "Ruida files"
    mime_types = ("application/x-rd-file", "application/octet-stream")
    extensions = (".rd",)

    def __init__(self, data: bytes):
        self.raw_data = data
        self._job_cache: Optional[RuidaJob] = None
        self._ops_cache: Optional[Ops] = None
        self._extents_cache: Optional[Tuple[float, float, float, float]] = None

    def _get_job(self) -> RuidaJob:
        """Parses the Ruida data and caches the resulting job."""
        if self._job_cache is None:
            parser = RuidaParser(self.raw_data)
            self._job_cache = parser.parse()
        return self._job_cache

    def _get_extents(self) -> Tuple[float, float, float, float]:
        """Gets the extents of the job, using a cache."""
        if self._extents_cache is None:
            job = self._get_job()
            self._extents_cache = job.get_extents()
        return self._extents_cache

    def get_vector_ops(self) -> Optional[Ops]:
        """
        Returns the parsed vector operations. The coordinate system is
        canonical (Y-up). Since Ruida data is Y-down, we flip it vertically.
        """
        if self._ops_cache is not None:
            return self._ops_cache

        job = self._get_job()
        if not job.commands:
            self._ops_cache = Ops()
            return self._ops_cache

        _min_x, min_y, _max_x, max_y = self._get_extents()
        y_flip_val = max_y + min_y

        ops = Ops()
        for cmd in job.commands:
            # Check the command type first, then safely access params.
            if cmd.command_type in ("Move_Abs", "Cut_Abs"):
                # Ensure params are valid before unpacking.
                if not cmd.params or len(cmd.params) != 2:
                    logger.warning(
                        f"Skipping Ruida command with invalid params: {cmd}"
                    )
                    continue

                x, y = cmd.params
                flipped_y = y_flip_val - y
                if cmd.command_type == "Move_Abs":
                    ops.move_to(x, flipped_y)
                elif cmd.command_type == "Cut_Abs":
                    ops.line_to(x, flipped_y)
        self._ops_cache = ops
        return self._ops_cache

    def get_natural_size(
        self, px_factor: float = 0.0
    ) -> Tuple[Optional[float], Optional[float]]:
        """Returns the natural size of the job in mm."""
        min_x, min_y, max_x, max_y = self._get_extents()
        width = max_x - min_x
        height = max_y - min_y
        return width, height

    def get_aspect_ratio(self) -> float:
        """Calculates the aspect ratio."""
        w, h = self.get_natural_size()
        if w and h and h > 0:
            return w / h
        return 1.0

    def render_to_pixels(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """
        Renders the vector paths to a Cairo surface for previews.
        """
        if width <= 0 or height <= 0:
            return None

        render_width, render_height = width, height
        # Downscale if requested size exceeds Cairo's limit
        if (
            render_width > CAIRO_MAX_DIMENSION
            or render_height > CAIRO_MAX_DIMENSION
        ):
            scale_factor = 1.0
            if render_width > CAIRO_MAX_DIMENSION:
                scale_factor = CAIRO_MAX_DIMENSION / render_width
            if render_height > CAIRO_MAX_DIMENSION:
                scale_factor = min(
                    scale_factor, CAIRO_MAX_DIMENSION / render_height
                )
            render_width = max(1, int(render_width * scale_factor))
            render_height = max(1, int(render_height * scale_factor))

        job = self._get_job()
        min_x, min_y, max_x, max_y = self._get_extents()

        surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32, render_width, render_height
        )
        ctx = cairo.Context(surface)
        ctx.set_source_rgba(0, 0, 0, 0)  # Transparent background
        ctx.paint()

        if not job.commands:
            return surface

        ctx.set_source_rgb(0, 0, 0)  # Black lines

        job_width_mm = max_x - min_x
        job_height_mm = max_y - min_y
        scale_x = render_width / job_width_mm if job_width_mm > 0 else 1
        scale_y = render_height / job_height_mm if job_height_mm > 0 else 1

        # Transform the Cairo context to match the Ruida data.
        # Ruida data has Y=0 at the top (Y-down), just like Cairo.
        # We only need to scale the job to fit the surface dimensions
        # and translate it so its top-left corner is at the origin.
        ctx.scale(scale_x, scale_y)
        ctx.translate(-min_x, -min_y)

        # Set line width in the new scaled space to ensure a 1px line
        # regardless of zoom level.
        effective_scale = min(scale_x, scale_y)
        if effective_scale > 0:
            ctx.set_line_width(1.0 / effective_scale)
        else:
            ctx.set_line_width(1.0)

        # Draw the paths
        for cmd in job.commands:
            # Check the command type and validate params before using them.
            if cmd.params and len(cmd.params) == 2:
                if cmd.command_type == "Move_Abs":
                    ctx.move_to(*cmd.params)
                elif cmd.command_type == "Cut_Abs":
                    # Ensure a path is started before drawing a line
                    try:
                        ctx.get_current_point()
                    except cairo.Error:
                        ctx.move_to(*cmd.params)
                    ctx.line_to(*cmd.params)
        ctx.stroke()

        return surface

    def _render_to_vips_image(
        self, width: int, height: int
    ) -> Optional[pyvips.Image]:
        """
        Renders to a cairo surface and converts it to a pyvips image.
        This is required for the base class's chunking mechanism.
        """
        surface = self.render_to_pixels(width, height)
        if not surface:
            return None

        buf = surface.get_data()
        bgra_img = pyvips.Image.new_from_memory(
            buf, surface.get_width(), surface.get_height(), 4, "uchar"
        )

        # The base renderer's chunking logic expects an RGBA image.
        # Cairo's ARGB32 format is BGRA in memory on little-endian systems.
        # We need to reorder the bands from BGRA to RGBA.
        b, g, r, a = bgra_img[0], bgra_img[1], bgra_img[2], bgra_img[3]
        rgba_img = r.bandjoin([g, b, a])

        return rgba_img
