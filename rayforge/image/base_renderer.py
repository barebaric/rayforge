from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
import logging
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    from ..core.source_asset_segment import SourceAssetSegment
    from ..core.geo import Geometry

logger = logging.getLogger(__name__)


class Renderer(ABC):
    """
    An abstract base class for any object that can render raw data to a
    pixel image. Renderers are stateless singletons.
    """

    @abstractmethod
    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> Optional[pyvips.Image]:
        """
        Renders raw data into a pyvips Image of the specified dimensions.
        This method performs the raw format conversion (e.g. SVG->Bitmap,
        PDF->Bitmap) but does NOT handle cropping, masking, or high-level
        caching, which are handled by the WorkPiece.

        Args:
            data: The raw bytes to render.
            width: The target pixel width.
            height: The target pixel height.
            **kwargs: Optional format-specific arguments (e.g. 'boundaries'
                      for vector renderers).

        Returns:
            A pyvips.Image, or None if rendering fails.
        """
        raise NotImplementedError

    @abstractmethod
    def get_natural_size_from_data(
        self,
        *,
        render_data: Optional[bytes],
        source_segment: Optional["SourceAssetSegment"],
        source_metadata: Optional[Dict[str, Any]],
        boundaries: Optional["Geometry"] = None,
        current_size: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Calculates the natural (intrinsic) size of the content in millimeters
        based on the provided data.
        """
        raise NotImplementedError
