from rayforge.image.image_util import convert_surface_to_grayscale_inplace
from .modifier import Modifier


class ToGrayscale(Modifier):
    """
    Removes colors from input surface.
    """

    def run(self, surface):
        convert_surface_to_grayscale_inplace(surface)
