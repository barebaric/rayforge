import inspect
from typing import Dict
from .base_importer import Importer
from .base_renderer import Renderer
from .shared.ops_renderer import OPS_RENDERER
from .dxf.importer import DxfImporter
from .dxf.renderer import DXF_RENDERER
from .pdf.importer import PdfImporter
from .pdf.renderer import PDF_RENDERER
from .png.importer import PngImporter
from .png.renderer import PNG_RENDERER
from .ruida.importer import RuidaImporter
from .ruida.renderer import RUIDA_RENDERER
from .svg.importer import SvgImporter
from .svg.renderer import SVG_RENDERER


def isimporter(obj):
    return (
        inspect.isclass(obj)
        and issubclass(obj, Importer)
        and obj is not Importer
    )


importers = [obj for name, obj in list(locals().items()) if isimporter(obj)]

importer_by_name = {imp.__name__: imp for imp in importers}

importer_by_mime_type = dict()
for base in importers:
    for mime_type in base.mime_types:
        importer_by_mime_type[mime_type] = base

importer_by_extension = dict()
for base in importers:
    for extension in base.extensions:
        importer_by_extension[extension] = base

renderer_by_name: Dict[str, Renderer] = {
    "DxfRenderer": DXF_RENDERER,
    "OpsRenderer": OPS_RENDERER,
    "PngRenderer": PNG_RENDERER,
    "PdfRenderer": PDF_RENDERER,
    "RuidaRenderer": RUIDA_RENDERER,
    "SvgRenderer": SVG_RENDERER,
}

renderer_by_importer_name: Dict[str, Renderer] = {
    "DxfImporter": DXF_RENDERER,
    "OpsRenderer": OPS_RENDERER,
    "PngImporter": PNG_RENDERER,
    "PdfImporter": PDF_RENDERER,
    "RuidaImporter": RUIDA_RENDERER,
    "SvgImporter": SVG_RENDERER,
}

__all__ = [
    "DxfImporter",
    "PdfImporter",
    "PngImporter",
    "RuidaImporter",
    "SvgImporter",
    "importers",
    "importer_by_name",
    "importer_by_mime_type",
    "importer_by_extension",
    "renderer_by_name",
    "renderer_by_importer_name",
]
