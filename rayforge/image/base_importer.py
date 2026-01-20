from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import enum
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING, Set
from ..core.vectorization_spec import PassthroughSpec, TraceSpec
from .assembler import ItemAssembler
from .engine import NormalizationEngine

if TYPE_CHECKING:
    from ..core.item import DocItem
    from ..core.vectorization_spec import VectorizationSpec
    from ..core.source_asset import SourceAsset
    from ..core.sketcher.sketch import Sketch
    from .structures import ParsingResult, VectorizationResult


class ImporterFeature(enum.Flag):
    """
    Defines the capabilities of an Importer class.
    """

    NONE = 0
    BITMAP_TRACING = enum.auto()
    DIRECT_VECTOR = enum.auto()
    LAYER_SELECTION = enum.auto()
    PROCEDURAL_GENERATION = enum.auto()


@dataclass
class LayerInfo:
    """
    A lightweight descriptor for a single layer discovered in a file scan.
    """

    id: str  # Machine-readable identifier (e.g., SVG group ID)
    name: str  # User-facing name for the layer
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


class Importer(ABC):
    """
    An abstract base class that defines the interface for all importers.

    An Importer acts as a factory, taking raw file data and producing a
    self-contained `ImportPayload`. This payload contains the `SourceAsset`
    (the link to the original file) and a list of `DocItem` objects
    (typically `WorkPiece` instances) ready to be added to a document.

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
    ) -> Optional[ImportPayload]:
        """
        Parses the raw data and returns a self-contained ImportPayload.
        This is the template method that orchestrates the import pipeline.
        """
        # 1. Parse
        parse_result = self.parse()
        if not parse_result:
            return None

        # 2. Create Source
        source_asset = self.create_source_asset(parse_result)

        # 3. Vectorize
        spec = vectorization_spec or PassthroughSpec()
        # For vector formats, if no layers with geometry were found,
        # return early with no items.
        if not parse_result.layers and not isinstance(spec, TraceSpec):
            return ImportPayload(source=source_asset, items=[])

        vec_result = self.vectorize(parse_result, spec)

        # 4. Layout
        engine = NormalizationEngine()
        plan = engine.calculate_layout(vec_result, spec)

        if not plan:
            return ImportPayload(source=source_asset, items=[])

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

        return ImportPayload(source_asset, items)
