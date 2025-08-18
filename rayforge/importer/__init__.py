# flake8: noqa:F401
import inspect
from .base import Importer
from .dxf import DxfImporter
from .opsimport import OpsImporter
from .pdf import PdfImporter
from .png import PngImporter
from .ruida import RuidaImporter
from .svg import SvgImporter

def isimporter(obj):
    return (inspect.isclass(obj)
            and issubclass(obj, Importer)
            and obj is not Importer)

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

__all__ = [
    "DxfImporter",
    "OpsImporter",
    "PdfImporter",
    "PngImporter",
    "RuidaImporter",
    "SvgImporter",
]
