import logging
import json
from typing import Optional, TYPE_CHECKING
import warnings
from ...core.sketcher import Sketch
from ..base_renderer import Renderer
from ..ops_renderer import OPS_RENDERER

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SketchRenderer(Renderer):
    """
    Renders a sketch by delegating its vector geometry to the OpsRenderer.
    """

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> Optional[pyvips.Image]:
        """
        Renders the sketch's vector data to a pyvips Image.
        It expects the geometry to be passed in the 'boundaries' kwarg.
        """
        logger.debug(
            f"SketchRenderer.render_base_image called. "
            f"width={width}, height={height}"
        )

        boundaries = kwargs.get("boundaries")
        has_boundaries_kwarg = boundaries and not boundaries.is_empty()

        # 1. Fallback parsing if boundaries are missing
        if not has_boundaries_kwarg:
            if not data:
                return pyvips.Image.black(width, height, bands=4)
            try:
                sketch_dict = json.loads(data.decode("utf-8"))
                sketch = Sketch.from_dict(sketch_dict)
                sketch.solve()
                boundaries = sketch.to_geometry()
            except Exception as e:
                logger.error(
                    f"Failed to render sketch from data: {e}", exc_info=True
                )
                return pyvips.Image.black(width, height, bands=4)

        if not boundaries or boundaries.is_empty():
            return pyvips.Image.black(width, height, bands=4)

        # 2. Delegate to the OPS_RENDERER
        return OPS_RENDERER.render_base_image(
            b"", width, height, boundaries=boundaries
        )


# Create a singleton instance for use by the importer
SKETCH_RENDERER = SketchRenderer()
