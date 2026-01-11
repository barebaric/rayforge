import asyncio
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
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ..context import get_context
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
    ImportPayload,
    bitmap_mime_types,
    import_file,
    import_file_from_bytes,
    importer_by_extension,
    importer_by_mime_type,
    importers,
)
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
    Contains the rendered image bytes and the document items to display.
    """

    image_bytes: bytes
    payload: Optional[ImportPayload]
    aspect_ratio: float = 1.0
    warnings: List[str] = field(default_factory=list)


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

    def analyze_import_target(
        self, file_path: Path, mime_type: Optional[str] = None
    ) -> ImportAction:
        """
        Analyzes a file path (and optional mime type) to determine how it
        should be imported.
        """
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file_path)

        # 1. Check if we have an importer for this file
        importer_cls = None
        if mime_type:
            importer_cls = importer_by_mime_type.get(mime_type)

        if not importer_cls and file_path.suffix:
            importer_cls = importer_by_extension.get(file_path.suffix.lower())

        if not importer_cls:
            return ImportAction.UNSUPPORTED

        # 2. Determine if it requires configuration (Interactive) or can load
        # directly. Raster images, SVGs, and PDFs usually require configuration
        # (tracing, layers, cropping)
        is_raster_like = (
            (mime_type and mime_type in bitmap_mime_types)
            or mime_type == "application/pdf"
            or mime_type == "image/svg+xml"
            or importer_cls.is_bitmap  # Fallback check on the class property
        )

        if is_raster_like:
            return ImportAction.INTERACTIVE_CONFIG

        return ImportAction.DIRECT_LOAD

    def scan_import_file(
        self, file_bytes: bytes, mime_type: str
    ) -> Dict[str, Any]:
        """
        Lightweight scan of a file to extract metadata without full processing.
        """
        result = {}
        if mime_type == "image/svg+xml":
            try:
                from ..image.svg.svgutil import extract_layer_manifest

                result["layers"] = extract_layer_manifest(file_bytes)
            except Exception as e:
                logger.warning(f"Failed to scan SVG layers: {e}")
        return result

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
        try:
            # 1. Get vector geometry and metadata
            payload = import_file_from_bytes(
                file_bytes,
                filename,
                mime_type,
                spec,
            )

            if not payload or not payload.items:
                return None

            # Helper to find reference workpiece for crop/size logic
            reference_wp = self._extract_first_workpiece(payload.items)
            if not reference_wp:
                return None

            # 2. Generate high-res base image
            vips_image = None
            if isinstance(spec, TraceSpec):
                # For tracing, use pre-rendered data (SVG trace) or raw bytes
                image_bytes = payload.source.base_render_data or file_bytes
                if not image_bytes:
                    return None

                full_image = pyvips.Image.new_from_buffer(image_bytes, "")

                # Apply cropping if importer specified it
                if (
                    reference_wp.source_segment
                    and reference_wp.source_segment.crop_window_px
                ):
                    x, y, w, h = map(
                        int, reference_wp.source_segment.crop_window_px
                    )
                    vips_image = full_image.crop(x, y, w, h)
                else:
                    vips_image = full_image

            elif isinstance(spec, PassthroughSpec):
                # For direct SVG, render vector data at high resolution
                if not payload.source.base_render_data:
                    return None

                vips_image = pyvips.Image.new_from_buffer(
                    payload.source.base_render_data, "", scale=4.0
                )

            if not vips_image:
                return None

            # 3. Create thumbnail
            aspect_ratio = (
                vips_image.width / vips_image.height
                if vips_image.height
                else 1.0
            )

            preview_vips = vips_image.thumbnail_image(
                preview_size_px,
                height=preview_size_px,
                size="both",
            )

            if isinstance(spec, TraceSpec) and spec.invert:
                preview_vips = preview_vips.flatten(
                    background=[255, 255, 255]
                ).invert()

            png_bytes = preview_vips.pngsave_buffer()

            return PreviewResult(
                image_bytes=png_bytes,
                payload=payload,
                aspect_ratio=aspect_ratio,
            )

        except Exception as e:
            logger.error(
                f"Failed to generate import preview: {e}", exc_info=True
            )
            return None

    def _extract_first_workpiece(
        self, items: List[DocItem]
    ) -> Optional[WorkPiece]:
        """Recursively extract the first WorkPiece from a list of items."""
        for item in items:
            if isinstance(item, WorkPiece):
                return item
            if isinstance(item, Layer):
                # Check direct children
                for child in item.children:
                    res = self._extract_first_workpiece([child])
                    if res:
                        return res
        return None

    async def _load_file_async(
        self,
        filename: Path,
        mime_type: Optional[str],
        vectorization_spec: Optional[VectorizationSpec],
    ) -> Optional[ImportPayload]:
        """
        Runs the blocking import function in a background thread and returns
        the resulting payload.
        """
        return await asyncio.to_thread(
            import_file, filename, mime_type, vectorization_spec
        )

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
        if position_mm:
            strategy = PositionAtStrategy(items=items, position_mm=position_mm)
            deltas = strategy.calculate_deltas()
            if deltas:
                # All items get the same delta matrix to move the group
                delta_matrix = next(iter(deltas.values()))
                for item in items:
                    # Pre-multiply to apply translation in world space
                    item.matrix = delta_matrix @ item.matrix

                target_x, target_y = position_mm
                logger.info(
                    f"Positioned {len(items)} imported item(s) at "
                    f"({target_x:.2f}, {target_y:.2f}) mm"
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

        # This wrapper adapts our clean async method to the TaskManager,
        # which expects a coroutine that accepts a 'ctx' argument.
        async def wrapper(ctx, fn, mt, vec_spec, pos_mm):
            try:
                # Update task message for UI feedback
                ctx.set_message(
                    _("Importing {filename}...").format(filename=filename.name)
                )

                # 1. Run blocking I/O and CPU work in a background thread.
                payload = await self._load_file_async(fn, mt, vec_spec)

                # 2. Validate the result.
                if not payload or not payload.items:
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
                        self._finalize_import_on_main_thread(
                            payload, fn, pos_mm, vec_spec
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
        """
        if not items:
            return None

        # Get the bbox of the first item to initialize the bounds.
        min_x, min_y, w, h = items[0].bbox
        max_x = min_x + w
        max_y = min_y + h

        # Expand the bounds with the bboxes of the other items.
        for item in items[1:]:
            ix, iy, iw, ih = item.bbox
            min_x = min(min_x, ix)
            min_y = min(min_y, iy)
            max_x = max(max_x, ix + iw)
            max_y = max(max_y, iy + ih)

        return min_x, min_y, max_x - min_x, max_y - min_y

    def _fit_and_center_imported_items(self, items: List[DocItem]):
        """
        Scales imported items to fit within machine boundaries if they are too
        large, preserving aspect ratio. Then, it centers the items in the
        workspace.
        """
        config = get_context().config
        if not config:
            return
        machine = config.machine
        if not machine:
            # Cannot scale or center if machine dimensions are unknown
            logger.warning(
                "Cannot fit/center imported items: machine dimensions unknown."
            )
            return

        bbox = self._calculate_items_bbox(items)
        if not bbox:
            return

        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        machine_w, machine_h = machine.dimensions

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

            for item in items:
                # Pre-multiply to apply the transform in world space
                item.matrix = transform_matrix @ item.matrix

            # After scaling, recalculate the bounding box for centering
            bbox = self._calculate_items_bbox(items)
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
            for item in items:
                # Pre-multiply to apply translation in world space
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
                # Calculate current bounding box to find the center
                current_bbox = self._calculate_items_bbox(items)
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
                for item in items:
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
