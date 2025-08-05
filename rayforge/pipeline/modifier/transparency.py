from ...shared.util.cairoutil import make_transparent
from .modifier import Modifier


class MakeTransparent(Modifier):
    """
    Makes white pixels transparent.
    """
    def run(self, surface):
        make_transparent(surface)
