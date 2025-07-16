from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple
import cairo


class Renderer(ABC):
    """
    An abstract base class that defines the interface for all renderers.

    Each concrete renderer instance is created for a specific piece of
    image data and is responsible for managing its own internal state and
    implementation details.
    """

    label: Optional[str] = None
    mime_types: Optional[Tuple[str, ...]] = None
    extensions: Optional[Tuple[str, ...]] = None

    @abstractmethod
    def __init__(self, data: bytes):
        """
        The constructor that all subclasses must implement. It is
        responsible for receiving the raw byte data and preparing it for
        all subsequent rendering operations.
        """
        pass

    @abstractmethod
    def get_natural_size(
        self, px_factor: float = 0.0
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Returns the natural (untransformed) size of the image in mm.

        If the source document uses pixel units, the px_factor is used
        to convert those dimensions to millimeters.
        """
        pass

    @abstractmethod
    def get_aspect_ratio(self) -> float:
        """
        Returns the natural (untransformed) aspect ratio of the image.
        """
        pass

    @abstractmethod
    def render_to_pixels(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """
        Renders the image to a Cairo surface of specific pixel dimensions.
        """
        pass

    @abstractmethod
    def render_chunk(
        self,
        width_px: int,
        height_px: int,
        chunk_width: int = 100000,
        chunk_height: int = 20,
        overlap_x: int = 1,
        overlap_y: int = 0,
    ) -> Generator[Tuple[cairo.ImageSurface, Tuple[float, float]], None, None]:
        """
        Renders the image to a Cairo surface in chunks.
        """
        pass
