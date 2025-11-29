import logging
from typing import Optional, TYPE_CHECKING
import warnings
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

        # 2. Delegate to the OPS_RENDERER
        return OPS_RENDERER.render_base_image(
            b"", width, height, boundaries=boundaries
        )


# Create a singleton instance for use by the importer
SKETCH_RENDERER = SketchRenderer()
