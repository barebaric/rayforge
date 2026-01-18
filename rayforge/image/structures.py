from dataclasses import dataclass
from typing import List, Tuple, Optional
from ..core.matrix import Matrix


@dataclass
class LayerGeometry:
    """
    Describes the geometric properties of a specific layer within a parsed
    file.
    All coordinates are in the file's Native Coordinate System.
    """

    layer_id: str

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
    # this represents the trimmed viewbox.
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


@dataclass
class LayoutItem:
    """
    A single instruction for the Assembler (Phase 3).
    Represents one resulting WorkPiece configuration.
    """

    # The ID of the layer(s) this item represents.
    layer_id: Optional[str]

    # Matrix to transform the normalized (0-1) WorkPiece geometry
    # into its final Physical World position/scale (in mm).
    world_matrix: Matrix

    # Matrix to transform Native Coordinates -> Unit Square (0-1).
    # Used to normalize vector geometry or map pixels to the unit square.
    normalization_matrix: Matrix

    # The subset of the original file (in Native Coords) this item represents.
    # (x, y, w, h). Used for cropping images or limiting vector scope.
    crop_window: Tuple[float, float, float, float]
