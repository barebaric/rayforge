from typing import Optional
from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import (
    VectorizationSpec,
    TraceSpec,
    PassthroughSpec,
)
from ..base_importer import (
    Importer,
    ImporterFeature,
)
from ..structures import (
    ParsingResult,
    VectorizationResult,
    ImportResult,
    ImportManifest,
)
from .svg_trace import SvgTraceImporter
from .svg_vector import SvgVectorImporter


import logging

logger = logging.getLogger(__name__)


class SvgImporter(Importer):
    """
    A Facade importer for SVG files.

    It routes the import request to either the Vector strategy (for path
    extraction) or the Trace strategy (for rendering and tracing bitmaps),
    depending on the provided VectorizationSpec.
    """

    label = "SVG files"
    mime_types = ("image/svg+xml",)
    extensions = (".svg",)
    features = {
        ImporterFeature.DIRECT_VECTOR,
        ImporterFeature.BITMAP_TRACING,
        ImporterFeature.LAYER_SELECTION,
    }

    def scan(self) -> ImportManifest:
        # Use Vector importer for scanning as it's lightweight/standard
        return SvgVectorImporter(self.raw_data, self.source_file).scan()

    def get_doc_items(
        self, vectorization_spec: Optional[VectorizationSpec] = None
    ) -> Optional[ImportResult]:
        """
        Delegates the full import process to the appropriate strategy.
        """
        spec_to_use = vectorization_spec
        # If no spec is provided, default to the vector strategy.
        if spec_to_use is None:
            spec_to_use = PassthroughSpec()

        if isinstance(spec_to_use, TraceSpec):
            logger.debug("SvgImporter: Delegating to SvgTraceImporter.")
            delegate = SvgTraceImporter(self.raw_data, self.source_file)
        else:
            # This is the direct vector import path.
            # If no layers are specified (e.g. from CLI), assume the user wants
            # all layers imported into the current document layer.
            if (
                isinstance(spec_to_use, PassthroughSpec)
                and not spec_to_use.active_layer_ids
            ):
                logger.debug(
                    "Empty PassthroughSpec detected in facade. "
                    "Scanning for all available layers."
                )
                manifest = self.scan()
                all_layer_ids = [layer.id for layer in manifest.layers]
                if all_layer_ids:
                    logger.debug(
                        f"Populating spec with all layers: {all_layer_ids}"
                    )
                    # Create a new spec object that matches the UI's default.
                    # This ensures the "merge" strategy is used in the engine.
                    spec_to_use = PassthroughSpec(
                        active_layer_ids=all_layer_ids,
                        create_new_layers=False,
                    )

            logger.debug("SvgImporter: Delegating to SvgVectorImporter.")
            delegate = SvgVectorImporter(self.raw_data, self.source_file)

        import_result = delegate.get_doc_items(spec_to_use)

        # --- DIAGNOSTIC LOGGING ---
        if (
            import_result
            and import_result.payload
            and import_result.payload.items
        ):
            from ...core.workpiece import WorkPiece
            from ...core.layer import Layer

            def count_workpieces(items):
                count = 0
                for item in items:
                    if isinstance(item, WorkPiece):
                        count += 1
                    elif isinstance(item, Layer):
                        count += count_workpieces(item.children)
                return count

            def check_for_geometry(items):
                for item in items:
                    if isinstance(item, WorkPiece):
                        if (
                            item.source_segment
                            and item.source_segment.pristine_geometry
                        ):
                            return True
                    elif isinstance(item, Layer):
                        if check_for_geometry(item.children):
                            return True
                return False

            item_count = len(import_result.payload.items)
            wp_count = count_workpieces(import_result.payload.items)

            has_geo_in_segment = check_for_geometry(
                import_result.payload.items
            )

            item_info = (
                f"{item_count} total items ({wp_count} WorkPieces). "
                f"Pristine geometry in segment: {has_geo_in_segment}"
            )
        elif import_result:
            item_info = "0 items."
        else:
            item_info = "None (import failed)."

        logger.debug(f"SvgImporter delegate returned result with: {item_info}")
        # --- END DIAGNOSTIC ---

        # If we have a result, ensure the facade's errors (if any were
        # collected before delegation) are merged, though usually facade
        # does little before delegation.
        if import_result:
            import_result.warnings.extend(self._warnings)
            import_result.errors.extend(self._errors)

        return import_result

    # These abstract methods must be implemented to satisfy the ABC contract,
    # but get_doc_items bypasses them in this facade.

    def parse(self) -> Optional[ParsingResult]:
        raise NotImplementedError(
            "SvgImporter is a facade; parse is delegated via get_doc_items"
        )

    def vectorize(
        self, parse_result: ParsingResult, spec: VectorizationSpec
    ) -> VectorizationResult:
        raise NotImplementedError(
            "SvgImporter is a facade; vectorize is delegated via get_doc_items"
        )

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        raise NotImplementedError(
            "SvgImporter is a facade; create_source_asset is delegated"
        )
