from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
from ..core.geo import Geometry
from ..core.matrix import Matrix

if TYPE_CHECKING:
    from ..core.item import DocItem
    from ..core.source_asset import SourceAsset
    from ..core.sketcher.sketch import Sketch


@dataclass
class LayerInfo:
    """
    A lightweight descriptor for a single layer discovered in a file scan.
    """

    id: str
    name: str
    color: Optional[Tuple[float, float, float]] = None
    default_active: bool = True


@dataclass
class ImportManifest:
    """
    The result of a file scan, describing the file's contents and structure
    without performing a full import.
    """

    layers: List[LayerInfo] = field(default_factory=list)
    natural_size_mm: Optional[Tuple[float, float]] = None
    title: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class ImportPayload:
    """
    A container for the complete result of a file import operation.
    It's a self-contained package ready for integration into a document.
    """

    source: "SourceAsset"
    items: List["DocItem"]
    sketches: List["Sketch"] = field(default_factory=list)


@dataclass
class LayerGeometry:
    """
    Describes the geometric properties of a specific layer within a parsed
    file.
    All coordinates are in the file's Native Coordinate System.
    """

    layer_id: str
    name: str

    # The tight bounding box (x, y, width, height) of the content on this
    # layer.
    # For Raster images: The non-transparent pixel area.
    # For Vector files: The geometric bounding box of vectors on this layer.
    content_bounds: Tuple[float, float, float, float]


@dataclass
class ParsingResult:
    """
    The result of Phase 1 (Parsing).
    Contains pure geometric facts about the file without any layout decisions.
    """

    # The total canvas/page size defined by the file (x, y, w, h).
    # e.g., SVG ViewBox, PDF MediaBox, or Image Dimensions. For trimmed files,
    # this represents the trimmed viewbox. All coordinates must be absolute.
    page_bounds: Tuple[float, float, float, float]

    # Multiplier to convert Native units to Millimeters.
    # e.g., for 96 DPI SVG, this is 25.4 / 96.
    native_unit_to_mm: float

    # True if the native coordinate system has Y pointing down (SVG, Images).
    # False if Y points up (DXF).
    is_y_down: bool

    # Detailed geometry per layer.
    layers: List[LayerGeometry]

    # The bounds of the original, untrimmed page, if available. This is used
    # as the frame of reference for positioning trimmed content.
    untrimmed_page_bounds: Optional[Tuple[float, float, float, float]] = None

    # True if the pristine_geometry is already relative to its own
    # content_bounds origin (e.g. for trimmed SVGs). False if it's in the
    # global native coordinate system (e.g. for DXF).
    geometry_is_relative_to_bounds: bool = False

    # True if the page_bounds represent a cropped-to-content view, and the
    # workpiece should be sized to these bounds, not the untrimmed bounds.
    is_cropped_to_content: bool = False


@dataclass
class VectorizationResult:
    """
    The result of Phase 2 (Vectorization).
    Contains the final vector geometry that will be used for layout and
    assembly.
    """

    # The final, generated vector geometry for each layer.
    geometries_by_layer: Dict[Optional[str], Geometry]

    # A reference to the original parse facts for context (e.g., page bounds).
    source_parse_result: ParsingResult


@dataclass
class LayoutItem:
    """
    A single instruction for the Assembler (Phase 3).
    Represents one resulting WorkPiece configuration.
    """

    # The ID of the layer(s) this item represents.
    layer_id: Optional[str]

    # The name of the layer(s) this item represents.
    layer_name: Optional[str]

    # Matrix to transform the normalized (0-1) WorkPiece geometry
    # into its final Physical World position/scale (in mm).
    world_matrix: Matrix

    # Matrix to transform Native Coordinates -> Unit Square (0-1).
    # Used to normalize vector geometry or map pixels to the unit square.
    normalization_matrix: Matrix

    # The subset of the original file (in Native Coords) this item represents.
    # (x, y, w, h). Used for cropping images or limiting vector scope.
    crop_window: Tuple[float, float, float, float]


@dataclass
class ImportResult:
    """
    The complete, rich result of a file import operation, containing both the
    final payload and the intermediate parsing facts for contextual use (like
    previews).
    """

    payload: ImportPayload
    parse_result: ParsingResult
