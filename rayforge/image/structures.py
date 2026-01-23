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

    This class is part of Phase 1 (Scan) of the import pipeline. It provides
    metadata about layers without requiring full parsing of the file content.

    Attributes:
        id: Unique identifier for the layer within the file.
        name: Human-readable name for the layer.
        color: Optional RGB color tuple (0-1 range) for display purposes.
        default_active: Whether this layer should be active by default.
        feature_count: Optional count of geometric features in this layer.
    """

    id: str
    name: str
    color: Optional[Tuple[float, float, float]] = None
    default_active: bool = True
    feature_count: Optional[int] = None


@dataclass
class ImportManifest:
    """
    The result of Phase 1 (Scan) of the import pipeline.

    Describes the file's contents and structure without performing a full
    import. This is used for UI previews and layer selection dialogs.

    Coordinate System:
    ------------------
    natural_size_mm: Physical dimensions in millimeters (mm, Y-Up).
                    This represents the natural size of the document as
                    defined by the source format.

    Attributes:
        layers: List of LayerInfo objects describing each layer.
        natural_size_mm: Optional (width, height) in mm for the document.
        title: Optional title or name for the document.
        warnings: List of non-critical warnings discovered during scan.
        errors: List of errors discovered during scan.

    Error Handling:
    ---------------
    Scan errors are collected here but do not prevent the scan from
    returning. The presence of errors indicates the file may have issues
    but some information was still extracted.
    """

    layers: List[LayerInfo] = field(default_factory=list)
    natural_size_mm: Optional[Tuple[float, float]] = None
    title: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class LayerGeometry:
    """
    Describes the geometric properties of a specific layer within a parsed
    file.

    Coordinate System:
    ------------------
    All coordinates are in the file's Native Coordinate System:
    - For SVG: user units as defined by the viewBox
    - For DXF: drawing units
    - For images: pixels
    Y-axis orientation depends on source format (see ParsingResult.is_y_down)

    Frame of Reference:
    ------------------
    content_bounds are absolute coordinates within the document's native
    coordinate space. They represent the tight bounding box of all content
    on this layer.

    Attributes:
        layer_id: Unique identifier for this layer.
        name: Human-readable name for this layer.
        content_bounds: Tight bounding box (x, y, width, height) in Native
                        Coordinates. For raster images, this is the non-
                        transparent pixel area. For vector files, this is
                        the geometric bounding box.
    """

    layer_id: str
    name: str
    content_bounds: Tuple[float, float, float, float]


@dataclass
class ParsingResult:
    """
    The result of Phase 2 (Parse) of the import pipeline.

    Contains pure geometric facts about the file without any layout decisions.

    Coordinate System:
    ------------------
    document_bounds: Native Coordinates (file-specific units)
    - For SVG: user units as defined by the viewBox
    - For DXF: drawing units
    - For images: pixels
    Y-axis orientation is specified by is_y_down flag.

    world_frame_of_reference: World Coordinates (mm, Y-Up)
    - Physical world coordinates in millimeters
    - Y-axis points UP (Y-Up convention)
    - Origin (0,0) is at the bottom-left
    - Used as stable reference for UI previews

    Frame of Reference:
    ------------------
    - document_bounds are absolute within the document's native space
    - untrimmed_document_bounds provides reference for Y-inversion
    - world_frame_of_reference provides the stable world coordinate frame

    Attributes:
        document_bounds: Total canvas/page size (x, y, width, height) in
                         Native Coordinates. For trimmed files, this is the
                         trimmed viewbox. All coordinates are absolute.
        native_unit_to_mm: Multiplier to convert Native units to mm.
                           e.g., for 96 DPI SVG, this is 25.4 / 96.
        is_y_down: True if native Y points down (SVG, images), False if Y
                   points up (DXF).
        layers: List of LayerGeometry describing each layer's geometry.
        world_frame_of_reference: Authoritative frame (x, y, w, h) in World
                                  Coordinates (mm, Y-Up) encompassing the
                                  entire import operation.
        background_world_transform: Matrix for positioning background image
                                    within world_frame_of_reference.
        untrimmed_document_bounds: Optional bounds of original untrimmed
                                    page. Used as reference for positioning
                                    trimmed content.
        geometry_is_relative_to_bounds: True if geometry is relative to
                                         content_bounds origin (trimmed
                                         SVGs). False if in global native
                                         coords (DXF).
        is_cropped_to_content: True if document_bounds represent a cropped-to-
                               content view and workpiece should be sized to
                               these bounds, not untrimmed bounds.

    Error Handling:
    ---------------
    This structure represents parsing results. Errors during parsing are
    collected by the Importer and returned in ImportResult, not here.
    """

    document_bounds: Tuple[float, float, float, float]
    native_unit_to_mm: float
    is_y_down: bool
    layers: List[LayerGeometry]
    world_frame_of_reference: Tuple[float, float, float, float]
    background_world_transform: Matrix
    untrimmed_document_bounds: Optional[Tuple[float, float, float, float]] = (
        None
    )
    geometry_is_relative_to_bounds: bool = False
    is_cropped_to_content: bool = False


@dataclass
class VectorizationResult:
    """
    The result of Phase 3 (Vectorize) of the import pipeline.

    Contains the final vector geometry that will be used for layout and
    assembly.

    Coordinate System:
    ------------------
    geometries_by_layer: Vector geometry in Native Coordinates
    - The geometry is still in the file's native coordinate system
    - It will be normalized by the NormalizationEngine during layout
    - Y-axis orientation matches the source format

    Frame of Reference:
    ------------------
    - Geometry coordinates are absolute within the document's native space
    - source_parse_result provides the context for coordinate transformations

    Attributes:
        geometries_by_layer: Final vector geometry for each layer. Keys are
                             layer IDs (None for single-layer content).
        source_parse_result: Reference to original ParsingResult for context
                            (e.g., page bounds, coordinate system info).
        fills_by_layer: Optional fill geometry per layer, primarily used by
                        the Sketch importer.

    Error Handling:
    ---------------
    This structure represents vectorization results. Errors during
    vectorization are collected by the Importer and returned in ImportResult.
    """

    geometries_by_layer: Dict[Optional[str], Geometry]
    source_parse_result: ParsingResult
    fills_by_layer: Dict[Optional[str], List[Geometry]] = field(
        default_factory=dict
    )


@dataclass
class LayoutItem:
    """
    A single instruction for Phase 5 (Assemble).

    Represents one resulting WorkPiece configuration as calculated by the
    NormalizationEngine.

    Coordinate System:
    ------------------
    world_matrix: Transforms from Normalized (0-1, Y-Up) to World (mm, Y-Up)
    - Input: Unit square coordinates (0,0) to (1,1), Y-Up
    - Output: Physical world coordinates in millimeters, Y-Up
    - Origin (0,0) is at the bottom-left of the workpiece

    normalization_matrix: Transforms from Native to Normalized (0-1, Y-Up)
    - Input: Native Coordinates (file-specific)
    - Output: Unit square coordinates (0,1), Y-Up
    - Handles Y-axis inversion for Y-Down sources

    crop_window: Native Coordinates (file-specific units)
    - Absolute coordinates within the original document
    - Used to specify which portion of the source to use

    Frame of Reference:
    ------------------
    - crop_window is absolute in the document's native coordinate space
    - world_matrix positions the workpiece in the world coordinate system
    - normalization_matrix handles the coordinate system conversion

    Attributes:
        layer_id: Optional ID of the layer(s) this item represents.
        layer_name: Optional human-readable name for the layer(s).
        world_matrix: Matrix transforming normalized (0-1) geometry to final
                      World position/scale (mm, Y-Up).
        normalization_matrix: Matrix transforming Native Coordinates to Unit
                             Square (0-1, Y-Up).
        crop_window: Subset of original file (x, y, w, h) in Native Coords.
                     Used for cropping images or limiting vector scope.
    """

    layer_id: Optional[str]
    layer_name: Optional[str]
    world_matrix: Matrix
    normalization_matrix: Matrix
    crop_window: Tuple[float, float, float, float]


@dataclass
class ImportPayload:
    """
    A container for the complete result of Phase 5 (Assemble).

    This is the final output of the import pipeline, containing a
    self-contained package ready for integration into a document.

    Coordinate System:
    ------------------
    All DocItems in this payload are already positioned in World
    Coordinates (mm, Y-Up) with their transformation matrices applied.

    Attributes:
        source: The SourceAsset representing the imported file.
        items: List of DocItems (WorkPieces or Layers) ready for insertion.
        sketches: Optional list of Sketch objects for special importers.

    Error Handling:
    ---------------
    This structure represents successful import results. Errors during
    import are handled in ImportResult, not here.
    """

    source: "SourceAsset"
    items: List["DocItem"]
    sketches: List["Sketch"] = field(default_factory=list)


@dataclass
class ImportResult:
    """
    The complete result of the five-phase import pipeline.

    Contains both the final payload and intermediate results for contextual
    use (like previews).

    Coordinate System:
    ------------------
    - payload: Contains DocItems in World Coordinates (mm, Y-Up)
    - parse_result: Contains Native Coordinates and World Coordinates
    - vectorization_result: Contains Native Coordinates

    Frame of Reference:
    ------------------
    - Each result maintains its own coordinate system as documented in
      the respective class docstrings
    - The world_frame_of_reference in parse_result provides the stable
      world coordinate frame for UI purposes

    Attributes:
        payload: Optional ImportPayload containing the final DocItems.
                 May be None if import failed completely.
        parse_result: Optional ParsingResult with geometric facts.
                      May be None if parsing failed.
        vectorization_result: Optional VectorizationResult with vector
                              geometry. May be None if vectorization was
                              not performed or failed.
        warnings: List of non-critical warnings collected during import.
                  These do not prevent the import from returning results.
        errors: List of errors collected during import. The presence of
                errors indicates the import may have partial or no results.

    Error Handling:
    ---------------
    This is the primary container for error reporting. Warnings indicate
    non-critical issues that were handled. Errors indicate problems that
    may have prevented complete import. The presence of errors does not
    necessarily mean the import failed completely - partial results may
    still be available in payload or intermediate results.
    """

    payload: Optional[ImportPayload]
    parse_result: Optional[ParsingResult]
    vectorization_result: Optional[VectorizationResult] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
