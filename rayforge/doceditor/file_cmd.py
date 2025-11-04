import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, cast, Callable

from ..core.item import DocItem
from ..core.layer import Layer
from ..core.matrix import Matrix
from ..core.vectorization_config import TraceConfig
from ..image import import_file, ImportPayload
from ..core.import_source import ImportSource
from ..context import get_context
from ..pipeline.artifact import JobArtifactHandle, JobArtifact
from ..undo import ListItemCommand
from .layout.align import PositionAtStrategy

if TYPE_CHECKING:
    from ..doceditor.editor import DocEditor
    from ..shared.tasker.manager import TaskManager


logger = logging.getLogger(__name__)


class FileCmd:
    """Handles file import and export operations."""

    def __init__(
        self,
        editor: "DocEditor",
        task_manager: "TaskManager",
    ):
        self._editor = editor
        self._task_manager = task_manager

    async def _run_importer_async(
        self,
        filename: Path,
        mime_type: Optional[str],
        vector_config: Optional[TraceConfig],
    ) -> Optional[ImportPayload]:
        """Runs the blocking import function in a background thread."""
        return await asyncio.to_thread(
            import_file, filename, mime_type, vector_config
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
        source: Optional[ImportSource],
        filename: Path,
    ):
        """
        Adds the imported items and their source to the document model using
        the history manager.
        """
        if source:
            self._editor.doc.add_import_source(source)

        target_layer = cast(Layer, self._editor.default_workpiece_layer)
        cmd_name = _(f"Import {filename.name}")

        with self._editor.history_manager.transaction(cmd_name) as t:
            for item in items:
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
        self._commit_items_to_document(payload.items, payload.source, filename)

    async def _load_file_async(
        self,
        filename: Path,
        mime_type: Optional[str],
        vector_config: Optional[TraceConfig],
        position_mm: Optional[Tuple[float, float]] = None,
    ):
        """
        The core awaitable logic for loading a file and committing it to the
        document. This is a pure async method without TaskManager context.

        Args:
            filename: Path to the file to import
            mime_type: MIME type of the file
            vector_config: Configuration for vectorization
            position_mm: Optional (x, y) tuple in world coordinates (mm)
                to center the imported item
        """
        try:
            # 1. Run blocking I/O and CPU work in a background thread.
            payload = await self._run_importer_async(
                filename, mime_type, vector_config
            )

            # 2. Validate the result.
            if not payload or not payload.items:
                if mime_type and mime_type.startswith("image/"):
                    msg = _(
                        f"Failed to import {filename.name}. The image file "
                        f"may be corrupted or in an unsupported format."
                    )
                else:
                    msg = _(
                        f"Import failed: No items were created "
                        f"from {filename.name}"
                    )
                logger.warning(
                    f"Importer created no items for '{filename.name}' "
                    f"(MIME: {mime_type})"
                )
                # Schedule the error notification on the main thread.
                self._task_manager.schedule_on_main_thread(
                    self._editor.notification_requested.send, self, message=msg
                )
                return

            # 3. Schedule the final positioning and committing on the main
            #    thread, and wait for it to complete.
            import_finished_event = asyncio.Event()

            def finalizer_wrapper():
                """Wraps the finalizer to signal completion."""
                try:
                    self._finalize_import_on_main_thread(
                        payload, filename, position_mm
                    )
                finally:
                    # Always set the event, even if finalization fails,
                    # to prevent the awaiter from hanging forever.
                    import_finished_event.set()

            self._task_manager.schedule_on_main_thread(finalizer_wrapper)

            # Wait here until the scheduled task has finished executing.
            await import_finished_event.wait()

        except Exception as e:
            logger.error(
                f"Import task for {filename.name} failed.", exc_info=e
            )
            # Re-raise the exception so the caller (e.g., the TaskManager)
            # knows about the failure.
            raise

    def load_file_from_path(
        self,
        filename: Path,
        mime_type: Optional[str],
        vector_config: Optional[TraceConfig],
        position_mm: Optional[Tuple[float, float]] = None,
    ):
        """
        Public, synchronous method to launch a file import in the background.
        This is the clean entry point for the UI.

        Args:
            filename: Path to the file to import
            mime_type: MIME type of the file
            vector_config: Configuration for vectorization
                (None for direct vector import)
            position_mm: Optional (x, y) tuple in world coordinates (mm)
                to center the imported item.
                        If None, items are centered on the workspace.
        """

        # This wrapper adapts our clean async method to the TaskManager,
        # which expects a coroutine that accepts a 'ctx' argument.
        async def wrapper(ctx, fn, mt, vc, pos_mm):
            try:
                # Update task message for UI feedback
                ctx.set_message(_(f"Importing {filename.name}..."))
                await self._load_file_async(fn, mt, vc, pos_mm)
                ctx.set_message(_("Import complete!"))
            except Exception:
                # The async method already logs the full error.
                # Just update the task status for the UI.
                ctx.set_message(_("Import failed."))
                # Re-raise to ensure the task manager marks the task as failed.
                raise

        self._task_manager.add_coroutine(
            wrapper,
            filename,
            mime_type,
            vector_config,
            position_mm,
            key=f"import-{filename}",
        )

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

        # Add a margin so items don't touch the edge.
        # Use 90% of the work area for fitting.
        margin_factor = 0.90
        effective_machine_w = machine_w * margin_factor
        effective_machine_h = machine_h * margin_factor

        # 1. Scale to fit if necessary, preserving aspect ratio
        scale_factor = 1.0
        if bbox_w > effective_machine_w or bbox_h > effective_machine_h:
            scale_w = effective_machine_w / bbox_w if bbox_w > 1e-9 else 1.0
            scale_h = effective_machine_h / bbox_h if bbox_h > 1e-9 else 1.0
            scale_factor = min(scale_w, scale_h)

        if scale_factor < 1.0:
            msg = _(
                "⚠️ Imported item was larger than the work area and has been "
                "scaled down to fit."
            )
            logger.info(msg)
            self._editor.notification_requested.send(
                self, message=msg, persistent=True
            )

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
                if artifact.gcode_bytes is None:
                    raise ValueError("Final artifact is missing G-code data.")

                gcode_str = artifact.gcode_bytes.tobytes().decode("utf-8")
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
