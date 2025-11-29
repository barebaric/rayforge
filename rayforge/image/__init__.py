import inspect
import logging
import mimetypes
from pathlib import Path
from typing import Dict, Optional, Type, Union, List
from ..core.vectorization_spec import VectorizationSpec
from ..core.workpiece import WorkPiece
from ..core.item import DocItem
from ..core.source_asset import SourceAsset
from .base_importer import Importer, ImportPayload
from .base_renderer import Renderer
from .bmp.importer import BmpImporter
from .bmp.renderer import BMP_RENDERER
from .dxf.importer import DxfImporter
from .dxf.renderer import DXF_RENDERER
from .jpg.importer import JpgImporter
from .jpg.renderer import JPG_RENDERER
from .material_test_grid_renderer import MaterialTestRenderer
from .ops_renderer import OPS_RENDERER
from .pdf.importer import PdfImporter
from .pdf.renderer import PDF_RENDERER
from .png.importer import PngImporter
from .png.renderer import PNG_RENDERER
from .procedural.renderer import PROCEDURAL_RENDERER
from .ruida.importer import RuidaImporter
from .ruida.renderer import RUIDA_RENDERER
from .sketch.importer import SketchImporter
from .sketch.renderer import SKETCH_RENDERER
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

bitmap_mime_types = set()
for base in importers:
    if base.is_bitmap:
        bitmap_mime_types.update(base.mime_types)


def _hydrate_workpieces_for_preview(
    items: List["DocItem"], source: "SourceAsset"
):
    """
    Recursively finds all WorkPieces in a list of items and attaches the
    transient renderer and data required for previews.
    """
    for item in items:
        if isinstance(item, WorkPiece):
            item._renderer = source.renderer
            # Set the transient data, preferring the processed version
            # (e.g., cropped SVG) if it exists.
            item._data = source.base_render_data or source.original_data

        # Recurse into children of containers (like Groups)
        if hasattr(item, "children") and item.children:
            _hydrate_workpieces_for_preview(item.children, source)


def import_file_from_bytes(
    file_data: bytes,
    source_file_name: str,
    mime_type: str,
    vectorization_spec: Optional[VectorizationSpec] = None,
) -> Optional[ImportPayload]:
    """
    Imports a file from raw byte data. Used for previews and in-memory
    operations where a file path is not available or desirable.

    Args:
        file_data: The raw bytes of the file.
        source_file_name: The original name of the file (for context).
        mime_type: The MIME type to determine the importer.
        vectorization_spec: An optional VectorizationSpec for vectorization.

    Returns:
        An ImportPayload or None on failure.
    """
    importer_class = importer_by_mime_type.get(mime_type)
    if not importer_class:
        logger.error(f"No importer found for MIME type: {mime_type}")
        return None

    try:
        source_file = Path(source_file_name)
        importer = importer_class(file_data, source_file=source_file)
        payload = importer.get_doc_items(vectorization_spec)

        # Hydrate the temporary WorkPiece(s) with a direct renderer AND data
        # link so they can be rendered without being part of a full document.
        if payload and payload.source:
            _hydrate_workpieces_for_preview(payload.items, payload.source)

        return payload
    except Exception as e:
        logger.error(
            f"Importer {importer_class.__name__} "
            f"failed for {source_file_name}",
            exc_info=e,
        )
        return None


def import_file(
    source: Union[Path, bytes],
    mime_type: Optional[str] = None,
    vectorization_spec: Optional[VectorizationSpec] = None,
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
        vectorization_spec: An optional VectorizationSpec for vectorization.

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
        return importer.get_doc_items(vectorization_spec)
    except Exception as e:
        logger.error(
            f"Importer {importer_class.__name__} failed for {source_file}",
            exc_info=e,
        )
        return None


renderer_by_name: Dict[str, Renderer] = {
    "BmpRenderer": BMP_RENDERER,
    "DxfRenderer": DXF_RENDERER,
    "ProceduralRenderer": PROCEDURAL_RENDERER,
    "JpgRenderer": JPG_RENDERER,
    "MaterialTestRenderer": MaterialTestRenderer(),
    "OpsRenderer": OPS_RENDERER,
    "PngRenderer": PNG_RENDERER,
    "PdfRenderer": PDF_RENDERER,
    "RuidaRenderer": RUIDA_RENDERER,
    "SvgRenderer": SVG_RENDERER,
    "SketchRenderer": SKETCH_RENDERER,
}

renderer_by_importer_name: Dict[str, Renderer] = {
    "BmpImporter": BMP_RENDERER,
    "DxfImporter": DXF_RENDERER,
    "JpgRenderer": JPG_RENDERER,
    "OpsRenderer": OPS_RENDERER,
    "PngImporter": PNG_RENDERER,
    "PdfImporter": PDF_RENDERER,
    "RuidaImporter": RUIDA_RENDERER,
    "SketchImporter": SKETCH_RENDERER,
    "SvgImporter": SVG_RENDERER,
}

__all__ = [
    "BmpImporter",
    "DxfImporter",
    "JpgImporter",
    "PdfImporter",
    "PngImporter",
    "RuidaImporter",
    "SketchImporter",
    "SvgImporter",
    "import_file",
    "import_file_from_bytes",
    "importers",
    "importer_by_name",
    "importer_by_mime_type",
    "importer_by_extension",
    "bitmap_mime_types",
    "renderer_by_name",
    "renderer_by_importer_name",
]
