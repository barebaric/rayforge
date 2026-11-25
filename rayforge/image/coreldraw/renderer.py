from ..dxf.renderer import DxfRenderer


class CorelDrawRenderer(DxfRenderer):
    """
    Renders CorelDRAW workpieces from their vectorized geometry.

    CorelDRAW drawings are drawn from the geometry the importer already
    extracts, so the rendering path is the one the DXF renderer uses.
    """


CDR_RENDERER = CorelDrawRenderer()
