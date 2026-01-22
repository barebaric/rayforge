from ..dxf.renderer import DxfRenderer


class RuidaRenderer(DxfRenderer):
    """
    A renderer for Ruida workpieces. Inherits vector rendering logic from
    DxfRenderer.
    """

    pass


# The RUIDA importer produces vector geometry, so it uses a renderer that
# can handle it. We create this alias for consistency and future extension.
RUIDA_RENDERER = RuidaRenderer()
