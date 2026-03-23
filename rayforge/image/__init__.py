import inspect
import logging
import mimetypes
from pathlib import Path
from typing import Optional, Type, Union, List
from ..core.vectorization_spec import VectorizationSpec, PassthroughSpec
from ..core.workpiece import WorkPiece
from ..core.item import DocItem
from ..core.source_asset import SourceAsset
from .base_importer import (
    Importer,
    ImporterFeature,
)
from .base_exporter import BaseExporter
from .structures import (
    ImportPayload,
    ImportResult,
    ParsingResult,
    ImportManifest,
    LayerInfo,
)
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
from .svg.importer import SvgImporter
from .svg.renderer import SVG_RENDERER
from .svg.exporter import GeometrySvgExporter
from .dxf.exporter import GeometryDxfExporter
from .registry import (
    exporter_registry,
    importer_registry,
    renderer_registry,
)

logger = logging.getLogger(__name__)


def isimporter(obj):
    return (
        inspect.isclass(obj)
        and issubclass(obj, Importer)
        and obj is not Importer
    )


for name, obj in list(locals().items()):
    if isimporter(obj):
        importer_registry.register(obj)


def isexporter(obj):
    return (
        inspect.isclass(obj)
        and issubclass(obj, BaseExporter)
        and obj is not BaseExporter
    )


for name, obj in list(locals().items()):
    if isexporter(obj):
        exporter_registry.register(obj)


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
    logger.debug(
        f"import_file_from_bytes: file_data_len={len(file_data)}, "
        f"source_file_name={source_file_name}, mime_type={mime_type}"
    )
    importer_class = importer_registry.get_by_mime_type(mime_type)
    if not importer_class:
        logger.error(f"No importer found for MIME type: {mime_type}")
        return None

    try:
        source_file = Path(source_file_name)
        importer = importer_class(file_data, source_file=source_file)

        # If no spec is given (e.g., initial preview), default to Passthrough
        spec_to_use = vectorization_spec or PassthroughSpec()

        import_result = importer.get_doc_items(spec_to_use)

        if not import_result:
            return None

        payload = import_result.payload
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
        importer_class = importer_registry.get_by_mime_type(mime_type)

    if not importer_class and isinstance(source, Path):
        file_extension = source.suffix.lower()
        if file_extension:
            importer_class = importer_registry.get_by_extension(file_extension)

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

    logger.debug(
        f"import_file: file_data_len={len(file_data)}, "
        f"source_file={source_file}, mime_type={mime_type}"
    )

    # 3. Execute importer
    try:
        importer = importer_class(file_data, source_file=source_file)
        import_result = importer.get_doc_items(vectorization_spec)
        # Unpack the result to return only the payload, maintaining the API
        return import_result.payload if import_result else None
    except Exception as e:
        logger.error(
            f"Importer {importer_class.__name__} failed for {source_file}",
            exc_info=e,
        )
        return None


_RENDERERS = [
    BMP_RENDERER,
    DXF_RENDERER,
    PROCEDURAL_RENDERER,
    JPG_RENDERER,
    MaterialTestRenderer(),
    OPS_RENDERER,
    PNG_RENDERER,
    PDF_RENDERER,
    RUIDA_RENDERER,
    SVG_RENDERER,
]

for renderer in _RENDERERS:
    renderer_registry.register(renderer)


def get_renderer_for_asset(asset_type: str) -> Optional[Renderer]:
    """Get the renderer for an asset type."""
    return renderer_registry.get(asset_type)


__all__ = [
    "BmpImporter",
    "DxfImporter",
    "JpgImporter",
    "PdfImporter",
    "PngImporter",
    "RuidaImporter",
    "SvgImporter",
    "ImporterFeature",
    "ImportManifest",
    "ImportPayload",
    "ImportResult",
    "ParsingResult",
    "LayerInfo",
    "import_file",
    "import_file_from_bytes",
    "exporter_registry",
    "GeometryDxfExporter",
    "GeometrySvgExporter",
    "importer_registry",
    "renderer_registry",
    "get_renderer_for_asset",
]
