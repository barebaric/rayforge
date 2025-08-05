# flake8: noqa:F401
import inspect
from .renderer import Renderer
from .dxf import DXFRenderer
from .pdf import PDFRenderer
from .png import PNGRenderer
from .svg import SVGRenderer

def isrenderer(obj):
    return (inspect.isclass(obj)
            and issubclass(obj, Renderer)
            and obj is not Renderer)

renderers = [obj for name, obj in list(locals().items()) if isrenderer(obj)]

renderer_by_mime_type = dict()
for renderer in renderers:
    for mime_type in renderer.mime_types:
        renderer_by_mime_type[mime_type] = renderer

renderer_by_extension = dict()
for renderer in renderers:
    for extension in renderer.extensions:
        renderer_by_extension[extension] = renderer

__all__ = [
    "DXFRenderer",
    "PDFRenderer",
    "PNGRenderer",
    "SVGRenderer",
]
