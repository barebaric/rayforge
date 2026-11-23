from ..dxf.renderer import DxfRenderer


class GcodeRenderer(DxfRenderer):
    """
    Renders G-code workpieces from their vector boundaries, like DXF.
    """


GCODE_RENDERER = GcodeRenderer()
