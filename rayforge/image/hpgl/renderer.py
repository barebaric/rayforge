from ..dxf.renderer import DxfRenderer


class HpglRenderer(DxfRenderer):
    """
    Renders HPGL workpieces from their vector boundaries, like DXF.
    """


HPGL_RENDERER = HpglRenderer()
