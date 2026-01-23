from __future__ import annotations
from abc import ABC, abstractmethod
import enum
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING, Set, List
from ..core.vectorization_spec import PassthroughSpec, TraceSpec
from .assembler import ItemAssembler
from .engine import NormalizationEngine

if TYPE_CHECKING:
    from ..core.vectorization_spec import VectorizationSpec
    from ..core.source_asset import SourceAsset
    from .structures import (
        ParsingResult,
        VectorizationResult,
        ImportPayload,
        ImportResult,
        ImportManifest,
    )


class ImporterFeature(enum.Flag):
    """
    Defines the capabilities of an Importer class.
    """

    NONE = 0
    BITMAP_TRACING = enum.auto()
    DIRECT_VECTOR = enum.auto()
    LAYER_SELECTION = enum.auto()
    PROCEDURAL_GENERATION = enum.auto()


class Importer(ABC):
    """
    An abstract base class that defines the interface for all importers.

    An Importer acts as a factory, taking raw file data and producing a
    self-contained `ImportResult`. This result contains the `ImportPayload`
    (the `SourceAsset` and `DocItem`s) and the `ParsingResult` (geometric
    facts used for contextual rendering).

    Five-Phase Import Pipeline:
    ---------------------------
    1. **Scan (Phase 1)**: Extract metadata without full processing.
       Returns ImportManifest with layer info, natural size, warnings/errors.

    2. **Parse (Phase 2)**: Extract geometric facts from the file.
       Returns ParsingResult with bounds, coordinate system info, layers.

    3. **Vectorize (Phase 3)**: Convert parsed data to vector geometry.
       Returns VectorizationResult with Geometry objects per layer.

    4. **Layout (Phase 4)**: Calculate transformations for positioning.
       NormalizationEngine produces LayoutItem with transformation matrices.

    5. **Assemble (Phase 5)**: Create final DocItems.
       ItemAssembler produces WorkPieces and Layers ready for insertion.

    Coordinate System Contract:
    --------------------------
    Importers must handle coordinate systems correctly:

    **Native Coordinates (Input/Output of parse/vectorize):**
    - File-specific coordinate system (SVG user units, DXF units, pixels)
    - Y-axis orientation varies by format
    - All bounds are absolute within the document's coordinate space
    - Units are converted to mm via native_unit_to_mm factor

    **World Coordinates (Final output):**
    - Physical world coordinates in millimeters (mm)
    - Y-axis points UP (Y-Up convention)
    - Origin (0,0) is at the bottom-left of the workpiece
    - All positions are absolute in the world coordinate system

    **Y-Down vs Y-Up:**
    - Y-Down formats (SVG, images): origin at top-left, Y increases downward
    - Y-Up formats (DXF): origin at bottom-left, Y increases upward
    - Importers must set is_y_down flag correctly in ParsingResult
    - NormalizationEngine handles Y-inversion for Y-Down sources

    Frame of Reference:
    ------------------
    - document_bounds are absolute in the document's native coordinate space
    - For Y-Down: origin is at top-left
    - For Y-Up: origin is at bottom-left
    - untrimmed_document_bounds provides reference for Y-inversion
    - world_frame_of_reference provides stable world coordinate frame

    Architectural Contract:
    -----------------------
    To prevent "double transformation" bugs, all importers MUST follow a
    strict separation of concerns between an object's intrinsic shape and
    its physical transformation in the document.

    1. **Generate Normalized Vectors**: The vector geometry created by the
       importer should represent the object's SHAPE, normalized to a standard
       unit size (e.g., fitting within a 1x1 box) while preserving the
       original aspect ratio.

    2. **Assign to WorkPiece**: This normalized `Geometry` is assigned to
       `WorkPiece.boundaries`. At this point, the `WorkPiece`'s transformation
       matrix should be the identity matrix (scale=1).

    3. **Apply Physical Size via Matrix**: The importer then determines the
       object's intended physical size in millimeters and calls
       `WorkPiece.set_size()`. This method correctly applies the physical
       dimensions by modifying the `WorkPiece.matrix`, scaling the
       normalized vectors to their final size.

    This ensures that the scale is applied only once, through the matrix,
    and that `WorkPiece.boundaries` remains a pure representation of shape.

    Error Handling Rules:
    --------------------
    **scan() and parse() methods:**
    - These methods COLLECT errors via add_error() and add_warning()
    - They must NEVER raise exceptions for expected error conditions
    - Errors are stored in self._errors and self._warnings lists
    - ImportManifest and ParsingResult include errors/warnings fields
    - The presence of errors does not prevent returning results

    **Other public methods (vectorize, create_source_asset, get_doc_items):**
    - These methods MAY raise exceptions for unexpected conditions
    - They should assume valid input from earlier phases
    - Errors during these phases are also collected via add_error()
    - The get_doc_items() template method handles error collection

    **General rules:**
    - Use add_warning() for non-critical issues that don't prevent import
    - Use add_error() for problems that may affect the result quality
    - Always return a result object (even if partial) rather than None
    - The ImportResult wrapper contains all collected errors/warnings
    """

    label: str
    mime_types: Tuple[str, ...]
    extensions: Tuple[str, ...]

    # The base set of features is empty. Subclasses MUST override this.
    features: Set[ImporterFeature] = set()

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        """
        The constructor that all subclasses must implement.
        """
        self.raw_data = data
        self.source_file = source_file or Path("Untitled")
        self._warnings: List[str] = []
        self._errors: List[str] = []

    def add_warning(self, message: str) -> None:
        """Records a warning message to be displayed to the user."""
        self._warnings.append(message)

    def add_error(self, message: str) -> None:
        """Records an error message to be displayed to the user."""
        self._errors.append(message)

    @abstractmethod
    def scan(self) -> ImportManifest:
        """
        Phase 1: Lightweight file scan.

        Extracts metadata and structural information without full processing.
        This method should be fast and avoid heavy computation like pixel
        processing or full geometry conversion.

        Coordinate System:
        ------------------
        natural_size_mm in returned ImportManifest should be in World
        Coordinates (mm, Y-Up) representing the document's physical size.

        Error Handling:
        ---------------
        This method COLLECTS errors via add_error() and add_warning().
        It must NEVER raise exceptions for expected error conditions.
        Errors are included in the returned ImportManifest.
        The presence of errors does not prevent returning a result.

        Returns:
            ImportManifest describing the file's contents, including layers,
            natural size, and any warnings/errors encountered.
        """
        raise NotImplementedError

    @abstractmethod
    def parse(self) -> Optional["ParsingResult"]:
        """
        Phase 2: Parse raw data into geometric facts.

        Extracts geometric information from the file including bounds,
        coordinate system details, and layer information.

        Coordinate System:
        ------------------
        Returned ParsingResult must have:
        - document_bounds: Native Coordinates (file-specific units)
        - is_y_down: True for Y-Down (SVG, images), False for Y-Up (DXF)
        - native_unit_to_mm: Conversion factor to millimeters
        - world_frame_of_reference: World Coordinates (mm, Y-Up)
        - layers: List of LayerGeometry with content_bounds in Native Coords

        Frame of Reference:
        ------------------
        - document_bounds are absolute in the document's native coordinate
            space
        - For Y-Down formats: origin at top-left
        - For Y-Up formats: origin at bottom-left
        - untrimmed_document_bounds provides reference for Y-inversion

        Error Handling:
        ---------------
        This method COLLECTS errors via add_error() and add_warning().
        It must NEVER raise exceptions for expected error conditions.
        Errors are stored in self._errors and self._warnings lists.
        The presence of errors does not prevent returning a result.

        Returns:
            ParsingResult containing geometric facts about the file,
            or None if parsing fails completely.
        """
        raise NotImplementedError

    @abstractmethod
    def vectorize(
        self, parse_result: "ParsingResult", spec: "VectorizationSpec"
    ) -> "VectorizationResult":
        """
        Phase 3: Convert parsed data to vector geometry.

        Converts the parsed data into vector Geometry objects according to the
        VectorizationSpec.

        Coordinate System:
        ------------------
        Returned VectorizationResult geometries_by_layer must be in Native
        Coordinates (file-specific units) as specified in parse_result.
        The NormalizationEngine will handle conversion to World Coordinates.

        Args:
            parse_result: The ParsingResult from the parse() method.
                         Contains coordinate system metadata (is_y_down,
                         native_unit_to_mm, etc.).
            spec: The VectorizationSpec describing how to vectorize.
                  Either PassthroughSpec (direct vector) or TraceSpec
                  (bitmap tracing).

        Returns:
            VectorizationResult containing the vectorized geometry per layer.
            The source_parse_result must reference the input parse_result.

        Error Handling:
        ---------------
        This method MAY raise exceptions for unexpected conditions.
        It should assume valid input from earlier phases.
        Errors can also be collected via add_error() for reporting.
        """
        raise NotImplementedError

    @abstractmethod
    def create_source_asset(
        self, parse_result: "ParsingResult"
    ) -> "SourceAsset":
        """
        Creates a SourceAsset representing the imported file.

        The SourceAsset provides access to the original file data for
        rendering and reference purposes.

        Args:
            parse_result: The ParsingResult from the parse() method.
                         Contains document bounds and coordinate system info.

        Returns:
            A SourceAsset for the imported file. The asset should store
            raw file data and any metadata needed for rendering.

        Error Handling:
        ---------------
        This method MAY raise exceptions for unexpected conditions.
        It should assume valid input from earlier phases.
        """
        raise NotImplementedError

    def get_doc_items(
        self, vectorization_spec: Optional["VectorizationSpec"] = None
    ) -> Optional["ImportResult"]:
        """
        Template method that orchestrates the full five-phase import pipeline.

        This method coordinates the complete import process:
        1. Parse (Phase 2)
        2. Create Source Asset
        3. Vectorize (Phase 3)
        4. Layout (Phase 4)
        5. Assemble (Phase 5)

        Coordinate System:
        ------------------
        The returned ImportResult contains:
        - payload: DocItems in World Coordinates (mm, Y-Up)
        - parse_result: Native and World Coordinates
        - vectorization_result: Native Coordinates

        Args:
            vectorization_spec: Optional VectorizationSpec. If None, a smart
                              default is chosen based on importer features
                              (PassthroughSpec for direct vector,
                              TraceSpec for bitmap tracing).

        Returns:
            ImportResult containing the final payload and all intermediate
            results. May contain partial results and errors if some phases
            failed. Returns None only in exceptional cases.

        Error Handling:
        ---------------
        This method collects errors from all phases and returns them in the
        ImportResult. It handles failures gracefully, returning partial
        results where possible.
        """
        # (Needed for downstream type hints)
        from .structures import (
            ImportPayload,
            ImportResult,
            VectorizationResult,
        )

        # 1. Parse
        parse_result = self.parse()
        if not parse_result:
            return ImportResult(
                payload=None,
                parse_result=None,
                warnings=self._warnings,
                errors=self._errors,
            )

        # 2. Create Source
        source_asset = self.create_source_asset(parse_result)

        # 3. Vectorize
        spec = vectorization_spec
        if not spec:
            # Smart default: Choose spec based on importer features.
            # Prefer direct vector if available, otherwise fall back to trace.
            if ImporterFeature.DIRECT_VECTOR in self.features:
                spec = PassthroughSpec()
            elif ImporterFeature.BITMAP_TRACING in self.features:
                spec = TraceSpec()
            else:
                # Fallback for importers that may not declare features yet
                spec = PassthroughSpec()

        # For vector formats, if no layers with geometry were found,
        # return early with no items. Only applies to TraceSpec since
        # PassthroughSpec's vectorize() has fallback logic for SVGs without
        # explicit layers.
        if not parse_result.layers and isinstance(spec, TraceSpec):
            return ImportResult(
                payload=ImportPayload(source=source_asset, items=[]),
                parse_result=parse_result,
                vectorization_result=VectorizationResult(
                    geometries_by_layer={}, source_parse_result=parse_result
                ),
                warnings=self._warnings,
                errors=self._errors,
            )

        vec_result = self.vectorize(parse_result, spec)

        # 4. Layout
        engine = NormalizationEngine()
        plan = engine.calculate_layout(vec_result, spec)

        if not plan:
            return ImportResult(
                payload=ImportPayload(source=source_asset, items=[]),
                parse_result=parse_result,
                vectorization_result=vec_result,
                warnings=self._warnings,
                errors=self._errors,
            )

        # 5. Assemble
        assembler = ItemAssembler()
        items = assembler.create_items(
            source_asset=source_asset,
            layout_plan=plan,
            spec=spec,
            source_name=self.source_file.stem,
            geometries=vec_result.geometries_by_layer,
            document_bounds=vec_result.source_parse_result.document_bounds,
        )

        payload = ImportPayload(source_asset, items)

        # Call the post-processing hook before returning the final result
        final_payload = self._post_process_payload(payload)

        # We return the parsing result from the vectorization phase, as it
        # may have been updated (e.g. bounds calculation for subset of layers)
        return ImportResult(
            payload=final_payload,
            parse_result=vec_result.source_parse_result,
            vectorization_result=vec_result,
            warnings=self._warnings,
            errors=self._errors,
        )

    def _post_process_payload(
        self, payload: "ImportPayload"
    ) -> "ImportPayload":
        """
        An optional hook for subclasses to modify the final payload after
        assembly. This is useful for importers that need to add extra data
        or links, like the SketchImporter.
        """
        return payload
