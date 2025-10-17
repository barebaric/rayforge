import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, cast, Dict, Callable
from dataclasses import asdict

from ..core.item import DocItem
from ..core.layer import Layer
from ..core.matrix import Matrix
from ..core.vectorization_config import TraceConfig
from ..image import import_file
from ..pipeline.jobrunner import (
    run_job_assembly_in_subprocess,
    JobDescription,
    WorkItemInstruction,
)
from ..pipeline.artifact.handle import ArtifactHandle
from ..pipeline.artifact.store import ArtifactStore
from ..undo import ListItemCommand

if TYPE_CHECKING:
    from ..doceditor.editor import DocEditor
    from ..config import ConfigManager
    from ..shared.tasker.manager import TaskManager
    from ..shared.tasker.task import Task


logger = logging.getLogger(__name__)


class FileCmd:
    """Handles file import and export operations."""

    def __init__(
        self,
        editor: "DocEditor",
        task_manager: "TaskManager",
        config_manager: "ConfigManager",
    ):
        self._editor = editor
        self._task_manager = task_manager
        self._config_manager = config_manager

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
            payload = await asyncio.to_thread(
                import_file, filename, mime_type, vector_config
            )

            if not payload or not payload.items:
                # Provide more specific error message based on file type
                if mime_type and mime_type.startswith("image/"):
                    msg = _(
                        "Failed to import {filename}. The image file may be "
                        "corrupted or in an unsupported format."
                    ).format(filename=filename.name)
                else:
                    msg = _(
                        "Import failed: No items were created "
                        "from {filename}"
                    ).format(filename=filename.name)
                logger.warning(
                    f"Importer created no items for '{filename.name}' "
                    f"(MIME: {mime_type})"
                )
                self._editor.notification_requested.send(self, message=msg)
                return

            # 2. Perform document modifications (must be on main thread).
            # In tests, the asyncio loop is the main thread, so this is safe.
            if payload.source:
                self._editor.doc.add_import_source(payload.source)

            imported_items = payload.items

            # Position items at specified location or centered on workspace
            if position_mm:
                self._position_imported_items_at(imported_items, position_mm)
            else:
                self._fit_and_center_imported_items(imported_items)

            target_layer = cast(Layer, self._editor.default_workpiece_layer)
            cmd_name = _("Import {name}").format(name=filename.name)

            with self._editor.history_manager.transaction(cmd_name) as t:
                for item in imported_items:
                    command = ListItemCommand(
                        owner_obj=target_layer,
                        item=item,
                        undo_command="remove_child",
                        redo_command="add_child",
                    )
                    t.execute(command)

        except Exception as e:
            logger.error(
                f"Import task for {filename.name} failed.", exc_info=e
            )
            # Re-raise the exception so the caller (e.g., the test) knows
            # about the failure.
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
            # The 'ctx' from the task manager is ignored.
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

    def _position_imported_items_at(
        self, items: List[DocItem], position_mm: Tuple[float, float]
    ):
        """
        Position imported items so their bounding box center is at the
        specified location.

        Args:
            items: List of imported DocItems
            position_mm: (x, y) tuple in world coordinates (mm)
                where items should be centered
        """
        machine = self._config_manager.config.machine
        if not machine:
            logger.warning(
                "Cannot position imported items: machine dimensions unknown."
            )
            return

        # Calculate the bounding box of all imported items
        bbox = self._calculate_items_bbox(items)
        if not bbox:
            return

        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        target_x, target_y = position_mm

        # Calculate current center of the bounding box
        current_center_x = bbox_x + bbox_w / 2
        current_center_y = bbox_y + bbox_h / 2

        # Calculate translation to move center to target position
        delta_x = target_x - current_center_x
        delta_y = target_y - current_center_y

        # Apply translation to all items
        if abs(delta_x) > 1e-9 or abs(delta_y) > 1e-9:
            translation_matrix = Matrix.translation(delta_x, delta_y)
            for item in items:
                # Pre-multiply to apply translation in world space
                item.matrix = translation_matrix @ item.matrix

            logger.info(
                f"Positioned {len(items)} imported item(s) at "
                f"({target_x:.2f}, {target_y:.2f}) mm"
            )

    def _fit_and_center_imported_items(self, items: List[DocItem]):
        """
        Scales imported items to fit within machine boundaries if they are too
        large, preserving aspect ratio. Then, it centers the items in the
        workspace.
        """
        machine = self._config_manager.config.machine
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
                "Imported item was larger than the work area and has been "
                "scaled down to fit."
            )
            logger.info(msg)
            self._editor.notification_requested.send(self, message=msg)

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

    def _prepare_job_description(self) -> JobDescription:
        """Constructs the JobDescription from the current document state."""
        doc = self._editor.doc
        ops_generator = self._editor.ops_generator
        machine = self._config_manager.config.machine
        if not machine:
            raise ValueError("Cannot prepare job: No machine configured.")

        work_items_by_step: Dict[str, list] = {}
        per_step_transformers_by_step: Dict[str, list] = {}

        for layer in doc.layers:
            if not layer.workflow:
                continue
            for step in layer.workflow.steps:
                work_items_for_step = []
                for item_step, workpiece in layer.get_renderable_items():
                    if item_step.uid != step.uid:
                        continue

                    handle = ops_generator.get_artifact_handle(
                        step.uid, workpiece.uid
                    )
                    if handle:
                        world_transform = workpiece.get_world_transform()
                        instruction = WorkItemInstruction(
                            artifact_handle_dict=handle.to_dict(),
                            world_transform_list=world_transform.to_list(),
                            workpiece_dict=workpiece.to_dict(),
                        )
                        work_items_for_step.append(instruction)

                if work_items_for_step:
                    work_items_by_step[step.uid] = work_items_for_step

                per_step_transformers_by_step[step.uid] = (
                    step.per_step_transformers_dicts
                )

        return JobDescription(
            work_items_by_step=work_items_by_step,
            per_step_transformers_by_step=per_step_transformers_by_step,
            machine_dict=machine.to_dict(),
            doc_dict=doc.to_dict(),
        )

    def assemble_job_in_background(
        self,
        when_done: Callable[
            [Optional[ArtifactHandle], Optional[Exception]], None
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
        try:
            job_desc = self._prepare_job_description()
        except Exception as e:
            when_done(None, e)
            return

        def _when_done_wrapper(task: "Task"):
            try:
                # Re-raises exceptions from the task
                handle_dict = task.result()
                handle = (
                    ArtifactHandle.from_dict(handle_dict)
                    if handle_dict
                    else None
                )
                when_done(handle, None)
            except Exception as e:
                when_done(None, e)

        self._task_manager.run_process(
            run_job_assembly_in_subprocess,
            job_description_dict=asdict(job_desc),
            key="job-assembly",
            when_done=_when_done_wrapper,
        )

    def export_gcode_to_path(self, file_path: Path):
        """
        Asynchronously generates and exports G-code to a specific path.
        This is a non-blocking, fire-and-forget method for the UI.
        """

        def _on_export_assembly_done(
            handle: Optional[ArtifactHandle], error: Optional[Exception]
        ):
            try:
                if error:
                    raise error
                if not handle:
                    raise ValueError("Assembly process returned no artifact.")

                # Get artifact, decode G-code, and write to file
                artifact = ArtifactStore.get(handle)
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
                    ArtifactStore.release(handle)

        self.assemble_job_in_background(when_done=_on_export_assembly_done)
