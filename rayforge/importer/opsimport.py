import logging
import cairo
from typing import Optional, Tuple, TYPE_CHECKING
from .base import Importer
from ..core.ops import Ops
from ..pipeline.encoder.cairoencoder import CairoEncoder

if TYPE_CHECKING:
    import pyvips

logger = logging.getLogger(__name__)


class OpsImporter(Importer):
    """A lightweight internal importer for an in-memory Ops object."""

    label = "Vector Data"
    mime_types = ("application/x-rayforge-ops",)
    extensions = tuple()

    def __init__(self, data: bytes, ops: Optional[Ops] = None):
        """Accepts either bytes (for the standard Importer API) or an Ops
        object directly for internal use."""
        self._ops = ops or Ops()

    def get_natural_size(
        self, px_factor: float = 0.0
    ) -> Tuple[Optional[float], Optional[float]]:
        if self._ops.is_empty():
            return 0.0, 0.0
        x1, y1, x2, y2 = self._ops.rect()
        return x2 - x1, y2 - y1

    def get_aspect_ratio(self) -> float:
        w, h = self.get_natural_size()
        return w / h if w and h else 1.0

    def render_to_pixels(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """Renders the internal Ops object to a Cairo surface."""
        if self._ops.is_empty() or width <= 0 or height <= 0:
            return None

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)
        ctx.set_source_rgba(0, 0, 0, 0)
        ctx.paint()

        ops_width, ops_height = self.get_natural_size()
        if (
            ops_width is None
            or ops_height is None
            or ops_width <= 0
            or ops_height <= 0
        ):
            return surface

        # THIS IS THE FIX: PREPARE THE OPS FOR THE ENCODER
        # The encoder expects Ops data in a machine-like coordinate system
        # (0,0 at bottom-left). Our internal ops are relative to their own
        # bounding box. We need to translate them.

        min_x, min_y, _, _ = self._ops.rect()
        # Create a translated copy of the ops so that their origin is at (0,0)
        translated_ops = self._ops.copy().translate(-min_x, -min_y)

        # Determine the scale factor to fit the (now translated) ops into
        # the target pixel dimensions.
        scale_x = width / ops_width
        scale_y = height / ops_height
        scale = min(scale_x, scale_y)

        # Center the final drawing on the surface
        scaled_w = ops_width * scale
        scaled_h = ops_height * scale
        trans_x = (width - scaled_w) / 2
        trans_y = (height - scaled_h) / 2
        ctx.translate(trans_x, trans_y)

        # The encoder will handle the rest of the scaling and Y-flipping.
        # We pass it the translated ops and the calculated uniform scale.
        encoder = CairoEncoder()
        encoder.encode(translated_ops, ctx, scale=(scale, scale))

        return surface

    def _render_to_vips_image(
        self, width: int, height: int
    ) -> Optional["pyvips.Image"]:
        return None

    def get_vector_ops(self) -> "Optional[Ops]":
        return self._ops.copy()
