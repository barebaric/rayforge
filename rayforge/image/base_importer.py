from __future__ import annotations
from abc import ABC, abstractmethod
import enum
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING, Set
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
    (the `SourceAsset` and `DocItem`s) and the `ParsingResult` (geometric facts
    used for contextual rendering).

    Architectural Contract:
    -----------------------
    To prevent "double transformation" bugs, all importers MUST follow a strict
    separation of concerns between an object's intrinsic shape and its
    physical transformation in the document.

    1.  **Generate Normalized Vectors**: The vector geometry created by the
        importer should represent the object's SHAPE, normalized to a standard
        unit size (e.g., fitting within a 1x1 box) while preserving the
        original aspect ratio.

    2.  **Assign to WorkPiece**: This normalized `Geometry` is assigned to
        `WorkPiece.boundaries`. At this point, the `WorkPiece`'s transformation
        matrix should be the identity matrix (scale=1).

    3.  **Apply Physical Size via Matrix**: The importer then determines the
        object's intended physical size in millimeters and calls
        `WorkPiece.set_size()`. This method correctly applies the physical
        dimensions by modifying the `WorkPiece.matrix`, scaling the
        normalized vectors to their final size.

    This ensures that the scale is applied only once, through the matrix,
    and that `WorkPiece.boundaries` remains a pure representation of shape.
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

    @abstractmethod
    def scan(self) -> ImportManifest:
        """
        Performs a lightweight scan of the file to extract metadata and
        structural information, like layers, without full processing.

        This method should be fast and avoid heavy computation like pixel
        processing or full geometry conversion.

        Returns:
            An ImportManifest describing the file's contents.
        """
        raise NotImplementedError

    @abstractmethod
    def parse(self) -> Optional["ParsingResult"]:
        """
        Parses the raw data and returns a ParsingResult.

        Returns:
            A ParsingResult containing geometric facts about the file,
            or None if parsing fails.
        """
        raise NotImplementedError

    @abstractmethod
    def vectorize(
        self, parse_result: "ParsingResult", spec: "VectorizationSpec"
    ) -> "VectorizationResult":
        """
        Vectorizes parsed data according to spec.

        Args:
            parse_result: The ParsingResult from the parse() method.
            spec: The VectorizationSpec describing how to vectorize.

        Returns:
            A VectorizationResult containing the vectorized geometry.
        """
        raise NotImplementedError

    @abstractmethod
    def create_source_asset(
        self, parse_result: "ParsingResult"
    ) -> "SourceAsset":
        """
        Creates a SourceAsset from the parse result.

        Args:
            parse_result: The ParsingResult from the parse() method.

        Returns:
            A SourceAsset for the imported file.
        """
        raise NotImplementedError

    def get_doc_items(
        self, vectorization_spec: Optional["VectorizationSpec"] = None
    ) -> Optional["ImportResult"]:
        """
        Parses the raw data and returns a self-contained ImportResult.
        This is the template method that orchestrates the import pipeline.
        """
        # (Needed for downstream type hints)
        from .structures import ImportPayload, ImportResult

        # 1. Parse
        parse_result = self.parse()
        if not parse_result:
            return None

        # 2. Create Source
        source_asset = self.create_source_asset(parse_result)

        # 3. Vectorize
        spec = vectorization_spec or PassthroughSpec()
        # For vector formats, if no layers with geometry were found,
        # return early with no items. Only applies to TraceSpec since
        # PassthroughSpec's vectorize() has fallback logic for SVGs without
        # explicit layers.
        if not parse_result.layers and isinstance(spec, TraceSpec):
            return ImportResult(
                payload=ImportPayload(source=source_asset, items=[]),
                parse_result=parse_result,
            )

        vec_result = self.vectorize(parse_result, spec)

        # 4. Layout
        engine = NormalizationEngine()
        plan = engine.calculate_layout(vec_result, spec)

        if not plan:
            return ImportResult(
                payload=ImportPayload(source=source_asset, items=[]),
                parse_result=parse_result,
            )

        # 5. Assemble
        assembler = ItemAssembler()
        items = assembler.create_items(
            source_asset=source_asset,
            layout_plan=plan,
            spec=spec,
            source_name=self.source_file.stem,
            geometries=vec_result.geometries_by_layer,
            page_bounds=vec_result.source_parse_result.page_bounds,
        )

        payload = ImportPayload(source_asset, items)

        # Call the post-processing hook before returning the final result
        final_payload = self._post_process_payload(payload)

        return ImportResult(payload=final_payload, parse_result=parse_result)

    def _post_process_payload(
        self, payload: "ImportPayload"
    ) -> "ImportPayload":
        """
        An optional hook for subclasses to modify the final payload after
        assembly. This is useful for importers that need to add extra data
        or links, like the SketchImporter.
        """
        return payload
