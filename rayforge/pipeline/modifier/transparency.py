from rayforge.image.image_util import make_surface_transparent
from .modifier import Modifier


class MakeTransparent(Modifier):
    """
    Makes white pixels transparent.
    """

    def run(self, surface):
        make_surface_transparent(surface)
