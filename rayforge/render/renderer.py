from abc import ABC, abstractmethod


class Renderer(ABC):
    """
    Reads image data and renders to a Cairo surface.
    """
    label = None
    mime_types = None
    extensions = None

    @classmethod
    def prepare(cls, data):
        """
        Called once for every image on import and can be used to preload
        or prepare the image.
        """
        return data

    @classmethod
    @abstractmethod
    def get_natural_size(cls, data, px_factor=0):
        """
        Returns the natural (untransformed) size of the image in mm, if
        known. Return None, None, otherwise.

        If px_factor is given and the document dimensions are in pixels,
        then the factor is used to convert to millimeters.
        If the document uses pixel unit and the factor is not given,
        a tuple of None is returned.
        """
        return None, None

    @classmethod
    @abstractmethod
    def get_aspect_ratio(cls, data):
        """
        Returns the natural (untransformed) aspect ratio of the image.
        """
        pass

    @classmethod
    @abstractmethod
    def render_workpiece(cls, data, width=None, height=None):
        """
        Renders to a Cairo surface.
        """
        pass

    @classmethod
    def render_chunk(cls,
                     data,
                     width_px,
                     height_px,
                     chunk_width=100000,
                     chunk_height=2,
                     overlap_x=1,
                     overlap_y=0):
        """
        Generator that renders to a Cairo surface, but in chunks.
        Yields one chunk per iteration.
        chunk_width and chunk_height are specified in pixels.
        """
        raise NotImplementedError
