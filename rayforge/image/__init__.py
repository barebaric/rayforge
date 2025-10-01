import inspect
import logging
import mimetypes
from pathlib import Path
from typing import Dict, Optional, Type, Union
from ..core.vectorization_config import TraceConfig
from .base_importer import Importer, ImportPayload
from .base_renderer import Renderer
from .bmp.importer import BmpImporter
from .bmp.renderer import BMP_RENDERER
from .dxf.importer import DxfImporter
from .dxf.renderer import DXF_RENDERER
from .jpg.importer import JpgImporter
from .jpg.renderer import JPG_RENDERER
from .ops_renderer import OPS_RENDERER
from .pdf.importer import PdfImporter
from .pdf.renderer import PDF_RENDERER
from .png.importer import PngImporter
from .png.renderer import PNG_RENDERER
from .ruida.importer import RuidaImporter
from .ruida.renderer import RUIDA_RENDERER
from .svg.importer import SvgImporter
from .svg.renderer import SVG_RENDERER


logger = logging.getLogger(__name__)


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


def import_file(
    source: Union[Path, bytes],
    mime_type: Optional[str] = None,
    vector_config: Optional[TraceConfig] = None,
) -> Optional[ImportPayload]:
    """
    A high-level convenience function to import a file from a path or raw
    data. It automatically determines the correct importer to use.

    The importer is chosen based on this priority:
    1. The provided `mime_type` override.
    2. The MIME type guessed from the filename (if `source` is a Path).
    3. The file extension (if `source` is a Path).

    Args:
        source: The pathlib.Path to the file or the raw bytes data.
        mime_type: An optional MIME type to force a specific importer.
        vector_config: An optional TraceConfig for vectorization.

    Returns:
        An ImportPayload containing the source and doc items, or None if
        the import fails or no suitable importer is found.
    """
    # If source is a path and no override is given, guess the MIME type.
    if isinstance(source, Path) and not mime_type:
        mime_type, _ = mimetypes.guess_type(source)

    # 1. Determine importer class
    importer_class: Optional[Type[Importer]] = None
    if mime_type:
        importer_class = importer_by_mime_type.get(mime_type)

    if not importer_class and isinstance(source, Path):
        file_extension = source.suffix.lower()
        if file_extension:
            importer_class = importer_by_extension.get(file_extension)

    if not importer_class:
        logger.error(f"No importer found for source: {source}")
        return None

    # 2. Prepare data and source path
    if isinstance(source, Path):
        source_file = source
        try:
            file_data = source.read_bytes()
        except IOError as e:
            logger.error(f"Could not read file {source}: {e}")
            return None
    else:  # is bytes
        source_file = Path("Untitled")
        file_data = source

    # 3. Execute importer
    try:
        importer = importer_class(file_data, source_file=source_file)
        return importer.get_doc_items(vector_config)
    except Exception as e:
        logger.error(
            f"Importer {importer_class.__name__} failed for {source_file}",
            exc_info=e,
        )
        return None


renderer_by_name: Dict[str, Renderer] = {
    "BmpRenderer": BMP_RENDERER,
    "DxfRenderer": DXF_RENDERER,
    "JpgRenderer": JPG_RENDERER,
    "OpsRenderer": OPS_RENDERER,
    "PngRenderer": PNG_RENDERER,
    "PdfRenderer": PDF_RENDERER,
    "RuidaRenderer": RUIDA_RENDERER,
    "SvgRenderer": SVG_RENDERER,
}

renderer_by_importer_name: Dict[str, Renderer] = {
    "BmpImporter": BMP_RENDERER,
    "DxfImporter": DXF_RENDERER,
    "JpgRenderer": JPG_RENDERER,
    "OpsRenderer": OPS_RENDERER,
    "PngImporter": PNG_RENDERER,
    "PdfImporter": PDF_RENDERER,
    "RuidaImporter": RUIDA_RENDERER,
    "SvgImporter": SVG_RENDERER,
}

__all__ = [
    "BmpImporter",
    "DxfImporter",
    "JpgImporter",
    "PdfImporter",
    "PngImporter",
    "RuidaImporter",
    "SvgImporter",
    "import_file",
    "importers",
    "importer_by_name",
    "importer_by_mime_type",
    "importer_by_extension",
    "renderer_by_name",
    "renderer_by_importer_name",
]
