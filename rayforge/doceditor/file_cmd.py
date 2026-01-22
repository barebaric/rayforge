import asyncio
import json
import logging
import mimetypes
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    cast,
    Set,
    Type,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ..context import get_context
from ..core.geo import Geometry
from ..core.item import DocItem
from ..core.layer import Layer
from ..core.matrix import Matrix
from ..core.source_asset import SourceAsset
from ..core.undo import ListItemCommand
from ..core.vectorization_spec import (
    PassthroughSpec,
    TraceSpec,
    VectorizationSpec,
)
from ..core.workpiece import WorkPiece
from ..image import (
    importer_by_extension,
    importer_by_mime_type,
    importers,
    ImporterFeature,
    ImportManifest,
    Importer,
)
from ..image.structures import ImportPayload, ImportResult, ParsingResult
from ..pipeline.artifact import JobArtifact, JobArtifactHandle
from .layout.align import PositionAtStrategy

if TYPE_CHECKING:
    from ..core.sketcher.sketch import Sketch
    from ..doceditor.editor import DocEditor
    from ..shared.tasker.manager import TaskManager


logger = logging.getLogger(__name__)


@dataclass
class PreviewResult:
    """
    Result of a preview generation operation.
    Contains the rendered image bytes, the document items to display, and the
    parsing context needed for correct rendering.
    """

    image_bytes: bytes
    payload: Optional[ImportPayload]
    parse_result: Optional[ParsingResult]  # Context for rendering
    aspect_ratio: float = 1.0
    warnings: List[str] = field(default_factory=list)
    content_bounds: Optional[Tuple[float, float, float, float]] = None


class ImportAction(Enum):
    """Determines the workflow required to import a specific file."""

    DIRECT_LOAD = auto()
    INTERACTIVE_CONFIG = auto()
    UNSUPPORTED = auto()


class FileCmd:
    """Handles file import and export operations."""

    def __init__(
        self,
        editor: "DocEditor",
        task_manager: "TaskManager",
    ):
        self._editor = editor
        self._task_manager = task_manager

    def get_supported_import_filters(self) -> List[Dict[str, Any]]:
        """
        Returns a list of dictionaries describing supported file types
        for UI dialogs.
        Each dict has 'label', 'extensions', and 'mime_types'.
        """
        filters = []
        for imp in importers:
            filters.append(
                {
                    "label": imp.label,
                    "extensions": imp.extensions,
                    "mime_types": imp.mime_types,
                }
            )
        return filters

    def get_importer_info(
        self, file_path: Path, mime_type: Optional[str]
    ) -> Tuple[Optional[Type[Importer]], Set[ImporterFeature]]:
        """
        Finds the importer for a file and returns its class and feature set.
        """
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file_path)

        importer_cls = None
        if mime_type:
            importer_cls = importer_by_mime_type.get(mime_type)

        if not importer_cls and file_path.suffix:
            importer_cls = importer_by_extension.get(file_path.suffix.lower())

        if importer_cls:
            return importer_cls, importer_cls.features
        return None, set()

    def analyze_import_target(
        self, file_path: Path, mime_type: Optional[str] = None
    ) -> ImportAction:
        """
        Analyzes a file path (and optional mime type) to determine how it
        should be imported.
        """
        importer_cls, features = self.get_importer_info(file_path, mime_type)

        if not importer_cls:
            return ImportAction.UNSUPPORTED

        # Any format that can be traced OR has selectable layers needs an
        # interactive dialog.
        if (
            ImporterFeature.BITMAP_TRACING in features
            or ImporterFeature.LAYER_SELECTION in features
        ):
            return ImportAction.INTERACTIVE_CONFIG

        return ImportAction.DIRECT_LOAD

    def scan_import_file(
        self, file_bytes: bytes, file_path: Path, mime_type: str
    ) -> ImportManifest:
        """
        Lightweight scan of a file to extract metadata without full processing.
        """
        importer_cls, _ = self.get_importer_info(file_path, mime_type)

        if not importer_cls:
            logger.warning(
                f"No importer found for mime type '{mime_type}' or "
                f"extension '{file_path.suffix}' during scan."
            )
            return ImportManifest(
                title=file_path.name,
                warnings=[f"Unsupported file type: {file_path.suffix}"],
            )

        try:
            importer_instance = importer_cls(
                data=file_bytes, source_file=file_path
            )
            manifest = importer_instance.scan()
            return manifest
        except Exception as e:
            logger.error(
                f"Error scanning file {file_path.name} with "
                f"{importer_cls.__name__}: {e}",
                exc_info=True,
            )
            return ImportManifest(
                title=file_path.name,
                warnings=[
                    "An unexpected error occurred during file analysis."
                ],
            )

    async def generate_preview(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        spec: VectorizationSpec,
        preview_size_px: int,
    ) -> Optional[PreviewResult]:
        """
        Generates a preview image and vector payload for the import dialog.
        Runs the heavy image processing in a background thread.
        """
        return await asyncio.to_thread(
            self._generate_preview_impl,
            file_bytes,
            filename,
            mime_type,
            spec,
            preview_size_px,
        )

    def _generate_preview_impl(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        spec: VectorizationSpec,
        preview_size_px: int,
    ) -> Optional[PreviewResult]:
        """Blocking implementation of preview generation."""
        importer_cls, _ = self.get_importer_info(Path(filename), mime_type)
        if not importer_cls:
            return None

        try:
            importer = importer_cls(
                data=file_bytes, source_file=Path(filename)
            )
            import_result = importer.get_doc_items(spec)

            if not import_result:
                return None

            # Even if no items were created, we might still be able to show a
            # preview of the source asset (e.g., an empty DXF).
            if not import_result.payload or not import_result.payload.items:
                logger.warning(
                    f"Import of '{filename}' produced no document items, "
                    "but attempting to generate a preview."
                )

            return self._generate_rich_preview_result(
                import_result, file_bytes, spec, preview_size_px
            )

        except Exception as e:
            logger.error(
                f"Failed to generate import preview: {e}", exc_info=True
            )
            return None

    def _generate_rich_preview_result(
        self,
        import_result: ImportResult,
        original_file_bytes: bytes,
        spec: VectorizationSpec,
        preview_size_px: int,
    ) -> Optional[PreviewResult]:
        """
        Generates the final PreviewResult from a rich ImportResult.
        This is the new central logic for creating preview bitmaps.
        """
        payload = import_result.payload
        parse_result = import_result.parse_result

        if not payload or not parse_result:
            return None

        renderer = payload.source.renderer
        if not renderer:
            return None

        # 1. Generate high-res base image for the background by delegating
        # to the source's specialized renderer.
        vips_image = None
        content_bounds = None
        target_dim = 2048  # Target for the longest edge of the hi-res preview

        _, _, w_native, h_native = parse_result.document_bounds
        if w_native <= 1e-9 or h_native <= 1e-9:
            # If there's no page size, we can't render a background.
            # This is not necessarily an error; a file might have vector
            # content but no defined canvas.
            pass
        else:
            aspect = w_native / h_native
            if aspect >= 1.0:
                render_width = target_dim
                render_height = max(1, int(target_dim / aspect))
            else:
                render_height = target_dim
                render_width = max(1, int(target_dim * aspect))

            vips_image = renderer.render_preview_image(
                import_result, render_width, render_height
            )

        # 2. Calculate content bounds for vector overlays from the
        # intermediate vectorization result.
        if import_result.vectorization_result:
            all_geos = Geometry()
            for geo in (
                import_result.vectorization_result.geometries_by_layer.values()
            ):
                if geo:
                    all_geos.extend(geo)

            if not all_geos.is_empty():
                min_x, min_y, max_x, max_y = all_geos.rect()
                content_bounds = (min_x, min_y, max_x, max_y)

        # 3. Create a thumbnail for the UI.
        if not vips_image:
            # If background rendering failed or was skipped, but we have
            # vectors, create a blank image to render the vectors on.
            if payload and payload.items:
                vips_image = pyvips.Image.black(
                    preview_size_px, preview_size_px
                )
            else:
                return None  # No background and no items, nothing to show.

        aspect_ratio = (
            vips_image.width / vips_image.height if vips_image.height else 1.0
        )
        preview_vips = vips_image.thumbnail_image(
            preview_size_px, height=preview_size_px, size="both"
        )

        if isinstance(spec, TraceSpec) and spec.invert:
            bands = preview_vips.bands
            if bands == 2:
                background = [255]
            elif bands == 4:
                background = [255, 255, 255]
            else:
                background = [255, 255, 255]
            preview_vips = preview_vips.flatten(background=background).invert()

        png_bytes = preview_vips.pngsave_buffer()

        return PreviewResult(
            image_bytes=png_bytes,
            payload=payload,
            parse_result=parse_result,
            aspect_ratio=aspect_ratio,
            content_bounds=content_bounds,
        )

    def _extract_first_workpiece(
        self, items: List[DocItem]
    ) -> Optional[WorkPiece]:
        """Recursively extract the first WorkPiece from a list of items."""
        for item in items:
            if isinstance(item, WorkPiece):
                return item
            if hasattr(item, "children"):
                res = self._extract_first_workpiece(item.children)
                if res:
                    return res
        return None

    async def _load_file_async(
        self,
        filename: Path,
        mime_type: Optional[str],
        vectorization_spec: Optional[VectorizationSpec],
    ) -> Optional[ImportResult]:
        """
        Runs the blocking import function in a background thread and returns
        the resulting rich ImportResult.
        """
        importer_cls, _ = self.get_importer_info(filename, mime_type)
        if not importer_cls:
            return None

        file_data = filename.read_bytes()
        importer = importer_cls(file_data, source_file=filename)
        return await asyncio.to_thread(
            importer.get_doc_items, vectorization_spec
        )

    def _get_positionable_content(self, items: List[DocItem]) -> List[DocItem]:
        """
        Extracts the actual content (WorkPieces, Groups) from a list of
        imported items, looking inside any top-level Layer containers.
        """
        content = []
        for item in items:
            if isinstance(item, Layer):
                content.extend(item.get_content_items())
            else:
                content.append(item)
        return content

    def _position_newly_imported_items(
        self,
        items: List[DocItem],
        position_mm: Optional[Tuple[float, float]],
    ):
        """
        Applies transformations to newly imported items, either positioning
        them at a specific point or fitting and centering them.
        This method modifies the items' matrices in-place.
        """
        logger.debug(
            f"_position_newly_imported_items: position_mm={position_mm}, "
            f"items={len(items)}"
        )

        # Get the actual content to be transformed, looking inside layers.
        content_to_transform = self._get_positionable_content(items)
        if not content_to_transform:
            return

        if position_mm:
            # Note: PositionAtStrategy needs the top-level items to calculate
            # the current group position correctly.
            strategy = PositionAtStrategy(items=items, position_mm=position_mm)
            deltas = strategy.calculate_deltas()
            if deltas:
                # All items get the same delta matrix to move the group
                delta_matrix = next(iter(deltas.values()))
                # Apply the delta to the actual content, not the containers.
                for item in content_to_transform:
                    item.matrix = delta_matrix @ item.matrix

                target_x, target_y = position_mm
                logger.info(
                    f"Positioned {len(content_to_transform)} imported "
                    f"item(s) at ({target_x:.2f}, {target_y:.2f}) mm"
                )
        else:
            self._fit_and_center_imported_items(items)

    def _commit_items_to_document(
        self,
        items: List[DocItem],
        source: Optional[SourceAsset],
        filename: Path,
        sketches: Optional[List["Sketch"]] = None,
        vectorization_spec: Optional[VectorizationSpec] = None,
    ):
        """
        Adds the imported items and their source to the document model using
        the history manager.
        """
        if source:
            self._editor.doc.add_asset(source)

        if sketches:
            for sketch in sketches:
                self._editor.doc.add_asset(sketch)

        target_layer = cast(Layer, self._editor.default_workpiece_layer)
        cmd_name = _("Import {filename}").format(filename=filename.name)

        create_new_layers = True
        if isinstance(vectorization_spec, PassthroughSpec):
            create_new_layers = vectorization_spec.create_new_layers

        with self._editor.history_manager.transaction(cmd_name) as t:
            for item in items:
                if isinstance(item, Layer) and create_new_layers:
                    owner = self._editor.doc
                    command = ListItemCommand(
                        owner_obj=owner,
                        item=item,
                        undo_command="remove_child",
                        redo_command="add_child",
                    )
                    t.execute(command)
                elif isinstance(item, Layer):
                    for child in item.get_content_items():
                        command = ListItemCommand(
                            owner_obj=target_layer,
                            item=child,
                            undo_command="remove_child",
                            redo_command="add_child",
                        )
                        t.execute(command)
                else:
                    command = ListItemCommand(
                        owner_obj=target_layer,
                        item=item,
                        undo_command="remove_child",
                        redo_command="add_child",
                    )
                    t.execute(command)

    def _finalize_import_on_main_thread(
        self,
        payload: ImportPayload,
        filename: Path,
        position_mm: Optional[Tuple[float, float]],
        vectorization_spec: Optional[VectorizationSpec] = None,
    ):
        """
        Performs the final steps of an import on the main thread.
        This includes positioning items (which may send UI notifications) and
        committing them to the document (which fires signals that update UI).
        """
        item_info = (
            f"{len(payload.items)} items"
            if payload and payload.items
            else "0 items"
        )
        logger.debug(f"Item_info: {item_info} position_mm: {position_mm}")
        # 1. Position the new items. This is now safe as it runs on the main
        #    thread, so any notifications it sends are valid.
        self._position_newly_imported_items(payload.items, position_mm)

        # 2. Add the positioned items to the document model. This is also
        #    safe now as all subsequent signal handling will be on the
        #    main thread.
        self._commit_items_to_document(
            payload.items,
            payload.source,
            filename,
            payload.sketches,
            vectorization_spec,
        )

    def load_file_from_path(
        self,
        filename: Path,
        mime_type: Optional[str],
        vectorization_spec: Optional[VectorizationSpec],
        position_mm: Optional[Tuple[float, float]] = None,
    ):
        """
        Public, synchronous method to launch a file import in the background.
        This is the clean entry point for the UI.

        Args:
            filename: Path to the file to import
            mime_type: MIME type of the file
            vectorization_spec: Configuration for vectorization
                (None for direct vector import)
            position_mm: Optional (x, y) tuple in world coordinates (mm)
                to center the imported item.
                        If None, items are centered on the workspace.
        """
        logger.debug(
            f"Loading file: {filename} "
            f"vectorization_spec: {vectorization_spec} "
            f"position_mm: {position_mm}"
        )

        # This wrapper adapts our clean async method to the TaskManager,
        # which expects a coroutine that accepts a 'ctx' argument.
        async def wrapper(ctx, fn, mt, vec_spec, pos_mm):
            try:
                # Update task message for UI feedback
                ctx.set_message(
                    _("Importing {filename}...").format(filename=filename.name)
                )

                # 1. Run blocking I/O and CPU work in a background thread.
                import_result = await self._load_file_async(fn, mt, vec_spec)

                # 2. Validate the result.
                if not import_result or not import_result.payload:
                    if mt and mt.startswith("image/"):
                        msg = _(
                            "Failed to import {filename}. The image file "
                            "may be corrupted or in an unsupported format."
                        ).format(filename=fn.name)
                    else:
                        msg = _(
                            "Import failed: No items were created "
                            "from {filename}"
                        ).format(filename=fn.name)
                    logger.warning(
                        f"Importer created no items for '{fn.name}' "
                        f"(MIME: {mt})"
                    )
                    # Schedule the error notification on the main thread.
                    self._task_manager.schedule_on_main_thread(
                        self._editor.notification_requested.send,
                        self,
                        message=msg,
                    )
                    ctx.set_message(_("Import failed."))
                    return

                # 3. Schedule finalization on main thread and wait for it to
                #    signal completion back to this (background) thread.
                loop = asyncio.get_running_loop()
                main_thread_done = loop.create_future()

                def finalizer_and_callback():
                    """Wraps finalizer to signal future on completion/error."""
                    try:
                        assert import_result.payload, "Missing import payload"
                        self._finalize_import_on_main_thread(
                            import_result.payload, fn, pos_mm, vec_spec
                        )
                        if not main_thread_done.done():
                            loop.call_soon_threadsafe(
                                main_thread_done.set_result, True
                            )
                    except Exception as e:
                        logger.error(
                            "Failed import finalization on main thread.",
                            exc_info=True,
                        )
                        if not main_thread_done.done():
                            loop.call_soon_threadsafe(
                                main_thread_done.set_exception, e
                            )

                self._task_manager.schedule_on_main_thread(
                    finalizer_and_callback
                )

                # Wait here until the main thread signals completion or error.
                await main_thread_done

                ctx.set_message(_("Import complete!"))
            except Exception as e:
                # This will catch failures from the importer or the finalizer.
                ctx.set_message(_("Import failed."))
                logger.error(
                    f"Import task for {fn.name} failed in wrapper.",
                    exc_info=e,
                )
                # Re-raise to ensure the task manager marks the task as failed.
                raise

        self._task_manager.add_coroutine(
            wrapper,
            filename,
            mime_type,
            vectorization_spec,
            position_mm,
            key=f"import-{filename}",
        )

    def execute_batch_import(
        self,
        files: List[Path],
        spec: VectorizationSpec,
        pos: Optional[Tuple[float, float]],
    ):
        """
        Imports multiple files using the same vectorization settings.
        This spawns individual import tasks for each file.
        """
        for file_path in files:
            # We assume files are valid if passed here, or guess mime type
            # individually
            mime_type, _ = mimetypes.guess_type(file_path)
            self.load_file_from_path(file_path, mime_type, spec, pos)

    def _calculate_items_bbox(
        self,
        items: List[DocItem],
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculates the world-space bounding box that encloses a list of
        DocItems by taking the union of their individual bboxes.
        This is more robust than item.bbox for un-parented items.
        """
        if not items:
            return None

        all_rects = []
        for item in items:
            # FIX: Use the item's matrix directly. This is robust for
            # items not yet in the document tree, as their matrix IS their
            # world transform at this point.
            item_transform = item.matrix
            item_bbox_local = item.get_local_bbox()

            if item_bbox_local:
                # Transform the four corners of the local bounding box
                corners = [
                    (item_bbox_local[0], item_bbox_local[1]),
                    (
                        item_bbox_local[0] + item_bbox_local[2],
                        item_bbox_local[1],
                    ),
                    (
                        item_bbox_local[0] + item_bbox_local[2],
                        item_bbox_local[1] + item_bbox_local[3],
                    ),
                    (
                        item_bbox_local[0],
                        item_bbox_local[1] + item_bbox_local[3],
                    ),
                ]
                world_corners = [
                    item_transform.transform_point(p) for p in corners
                ]

                min_x = min(p[0] for p in world_corners)
                min_y = min(p[1] for p in world_corners)
                max_x = max(p[0] for p in world_corners)
                max_y = max(p[1] for p in world_corners)
                all_rects.append((min_x, min_y, max_x - min_x, max_y - min_y))

        if not all_rects:
            return None

        # Calculate the union of all collected rectangles
        min_x, min_y, w, h = all_rects[0]
        max_x = min_x + w
        max_y = min_y + h

        for x, y, w, h in all_rects[1:]:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        return min_x, min_y, max_x - min_x, max_y - min_y

    def _fit_and_center_imported_items(self, items: List[DocItem]):
        """
        Scales imported items to fit within machine boundaries if they are too
        large, preserving aspect ratio. Then, it centers the items in the
        workspace.
        """
        config = get_context().config
        if not config or not config.machine:
            logger.warning(
                "Cannot fit/center imported items: machine dimensions unknown."
            )
            return

        # We must operate on the actual content (WorkPieces, Groups), not the
        # top-level containers (Layers).
        content_items = self._get_positionable_content(items)
        if not content_items:
            logger.warning("No positionable content found to fit/center.")
            return

        machine = config.machine
        # Calculate the bounding box of the actual content.
        bbox = self._calculate_items_bbox(content_items)
        if not bbox:
            logger.warning(
                "Cannot fit/center imported items: no bounding box."
            )
            return

        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        machine_w, machine_h = machine.dimensions
        logger.debug(
            f"_fit_and_center_imported_items: bbox=({bbox_x:.2f}, "
            f"{bbox_y:.2f}, {bbox_w:.2f}, {bbox_h:.2f}), "
            f"machine=({machine_w:.2f}, {machine_h:.2f})"
        )

        # 1. Scale to fit if necessary, preserving aspect ratio
        scale_factor = 1.0
        if bbox_w > machine_w or bbox_h > machine_h:
            scale_w = machine_w / bbox_w if bbox_w > 1e-9 else 1.0
            scale_h = machine_h / bbox_h if bbox_h > 1e-9 else 1.0
            scale_factor = min(scale_w, scale_h)

        if scale_factor < 1.0:
            # The pivot for scaling should be the center of the bounding box
            bbox_center_x = bbox_x + bbox_w / 2
            bbox_center_y = bbox_y + bbox_h / 2

            # The transformation is: T(pivot) @ S(scale) @ T(-pivot)
            t_to_origin = Matrix.translation(-bbox_center_x, -bbox_center_y)
            s = Matrix.scale(scale_factor, scale_factor)
            t_back = Matrix.translation(bbox_center_x, bbox_center_y)
            transform_matrix = t_back @ s @ t_to_origin

            # Apply the group transform to each piece of content.
            for item in content_items:
                item.matrix = transform_matrix @ item.matrix

            # After scaling, recalculate the bounding box for centering
            bbox = self._calculate_items_bbox(content_items)
            if not bbox:
                return  # Should not happen, but for safety
            bbox_x, bbox_y, bbox_w, bbox_h = bbox

        # 2. Center the (possibly scaled) items
        # Calculate translation to move bbox center to the machine center
        delta_x = (machine_w / 2) - (bbox_x + bbox_w / 2)
        delta_y = (machine_h / 2) - (bbox_y + bbox_h / 2)

        # Apply the same translation to all top-level imported items
        if abs(delta_x) > 1e-9 or abs(delta_y) > 1e-9:
            translation_matrix = Matrix.translation(delta_x, delta_y)
            # Apply the group transform to each piece of content.
            for item in content_items:
                item.matrix = translation_matrix @ item.matrix

        # 3. Notification with Undo logic
        # We define this after centering so the callback can handle the
        # final position correctly.
        if scale_factor < 1.0:

            def _undo_scaling_callback():
                """
                Reverts the auto-scaling applied during import.
                It scales the items back up around their CURRENT center.
                """
                # Use the content items for calculation and transformation
                current_bbox = self._calculate_items_bbox(content_items)
                if not current_bbox:
                    return

                cur_x, cur_y, cur_w, cur_h = current_bbox
                cur_cx = cur_x + cur_w / 2
                cur_cy = cur_y + cur_h / 2

                inv_scale = 1.0 / scale_factor

                # Create a matrix that scales by 1/factor around the current
                # center
                undo_matrix = Matrix.scale(
                    inv_scale, inv_scale, center=(cur_cx, cur_cy)
                )

                changes = []
                for item in content_items:
                    current = item.matrix
                    new_m = undo_matrix @ current
                    changes.append((item, current, new_m))

                self._editor.transform.create_transform_transaction(changes)

            msg = _(
                "⚠️ Imported item was larger than the work area and has been "
                "scaled down to fit."
            )
            logger.info(msg)
            self._editor.notification_requested.send(
                self,
                message=msg,
                persistent=True,
                action_label=_("Reset"),
                action_callback=_undo_scaling_callback,
            )

    def assemble_job_in_background(
        self,
        when_done: Callable[
            [Optional[JobArtifactHandle], Optional[Exception]], None
        ],
    ):
        """
        Asynchronously runs the full job assembly in a background process.
        This method is non-blocking and returns immediately.

        Args:
            when_done: A callback executed upon completion. It receives
                       an ArtifactHandle on success, or (None, error) on
                       failure.
        """
        self._editor.pipeline.generate_job_artifact(when_done=when_done)

    def export_gcode_to_path(self, file_path: Path):
        """
        Asynchronously generates and exports G-code to a specific path.
        This is a non-blocking, fire-and-forget method for the UI.
        """
        artifact_store = get_context().artifact_store

        def _on_export_assembly_done(
            handle: Optional[JobArtifactHandle], error: Optional[Exception]
        ):
            try:
                if error:
                    raise error
                if not handle:
                    raise ValueError("Assembly process returned no artifact.")

                # Get artifact, decode G-code, and write to file
                artifact = artifact_store.get(handle)
                if not isinstance(artifact, JobArtifact):
                    raise ValueError("Expected a JobArtifact for export.")
                if artifact.machine_code_bytes is None:
                    raise ValueError("Final artifact is missing G-code data.")

                gcode_str = artifact.machine_code_bytes.tobytes().decode(
                    "utf-8"
                )
                file_path.write_text(gcode_str, encoding="utf-8")

                logger.info(f"Successfully exported G-code to {file_path}")
                msg = _("Export successful: {name}").format(
                    name=file_path.name
                )
                self._editor.notification_requested.send(self, message=msg)

            except Exception as e:
                logger.error(
                    f"G-code export to {file_path} failed.", exc_info=e
                )
                self._editor.notification_requested.send(
                    self, message=_("Export failed: {error}").format(error=e)
                )
            finally:
                if handle:
                    artifact_store.release(handle)

        self.assemble_job_in_background(when_done=_on_export_assembly_done)

    def save_project_to_path(self, file_path: Path):
        """
        Saves the current document to a .ryp project file.
        This is a synchronous method for the UI.
        """
        try:
            doc_dict = self._editor.doc.to_dict()
            file_path.write_text(
                json.dumps(doc_dict, indent=2), encoding="utf-8"
            )
            self._editor.set_file_path(file_path)
            self._editor.mark_as_saved()
            logger.info(f"Successfully saved project to {file_path}")
            msg = _("Project saved: {name}").format(name=file_path.name)
            self._editor.notification_requested.send(self, message=msg)
            return True
        except Exception as e:
            logger.error(f"Failed to save project to {file_path}", exc_info=e)
            self._editor.notification_requested.send(
                self, message=_("Save failed: {error}").format(error=str(e))
            )
            return False

    def load_project_from_path(self, file_path: Path):
        """
        Loads a .ryp project file and replaces the current document.
        This is a synchronous method for the UI.
        """
        try:
            if not file_path.exists():
                msg = _("File not found: {name}").format(name=file_path.name)
                self._editor.notification_requested.send(self, message=msg)
                return False

            file_content = file_path.read_text(encoding="utf-8")
            doc_dict = json.loads(file_content)

            from ..core.doc import Doc

            new_doc = Doc.from_dict(doc_dict)

            self._editor.set_doc(new_doc)
            self._editor.set_file_path(file_path)
            self._editor.mark_as_saved()
            self._editor.doc.updated.send(self._editor.doc)

            logger.info(f"Successfully loaded project from {file_path}")
            msg = _("Project loaded: {name}").format(name=file_path.name)
            self._editor.notification_requested.send(self, message=msg)
            return True
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse project file {file_path}: {e}",
                exc_info=e,
            )
            self._editor.notification_requested.send(
                self, message=_("Invalid project file format")
            )
            return False
        except Exception as e:
            logger.error(
                f"Failed to load project from {file_path}", exc_info=e
            )
            self._editor.notification_requested.send(
                self, message=_("Load failed: {error}").format(error=str(e))
            )
            return False
