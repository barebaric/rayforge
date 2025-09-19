import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, cast
from ..core.item import DocItem
from ..core.matrix import Matrix
from ..importer import importer_by_mime_type, importer_by_extension
from ..undo import ListItemCommand
from ..shared.tasker.context import ExecutionContext
from ..pipeline.job import generate_job_ops
from ..pipeline.encoder.gcode import GcodeEncoder
from ..core.vectorization_config import TraceConfig
from ..core.import_source import ImportSource
from ..core.workpiece import WorkPiece
from ..core.layer import Layer

if TYPE_CHECKING:
    from .editor import DocEditor
    from ..shared.tasker.manager import TaskManager
    from ..config import ConfigManager


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

    def load_file_from_path(
        self,
        filename: Path,
        mime_type: Optional[str],
        vector_config: Optional[TraceConfig],
        when_done: Optional[Callable] = None,
    ):
        """
        Orchestrates the loading of a specific file path using the
        importer, running the import process in a background task.
        """

        async def import_coro(context: ExecutionContext):
            # Find importer class
            importer_class = None
            if mime_type:
                importer_class = importer_by_mime_type.get(mime_type)
            if not importer_class:
                file_extension = filename.suffix.lower()
                if file_extension:
                    importer_class = importer_by_extension.get(file_extension)

            if not importer_class:
                msg = f"No importer found for '{filename.name}'"
                logger.error(msg)
                context.set_message(f"Error: {msg}")
                raise ValueError(msg)

            context.set_message(_(f"Importing {filename.name}..."))
            context.flush()

            # 1. Read file data (blocking I/O)
            file_data = await asyncio.to_thread(filename.read_bytes)

            # 2. Instantiate importer and get items (potentially CPU-bound)
            def do_import_sync():
                importer = importer_class(file_data, source_file=filename)
                return importer.get_doc_items(vector_config)

            imported_items = await asyncio.to_thread(do_import_sync)

            if not imported_items:
                logger.warning(
                    f"Importer created no items for '{filename.name}'."
                )
                context.set_message(_("Import failed: No items were created."))
                return None  # Return None to signify no items to add.

            # 3. Create, register, and link the import source.
            import_source = ImportSource(
                source_file=filename,
                data=file_data,
                vector_config=vector_config,
            )

            # Return a tuple of the results for the main thread to handle.
            context.set_message(_("Import complete!"))
            context.set_progress(1.0)
            context.flush()
            return imported_items, import_source

        # This callback runs on the MAIN THREAD after the background task is
        # finished. It is the only safe place to modify the document.
        def commit_to_document(task):
            # First, call the original when_done if it exists.
            if when_done:
                when_done(task)

            try:
                # This will re-raise any exception from the background task.
                result = task.result()
                if not result:
                    return  # No items were imported.

                imported_items, import_source = result

                # 1. Register the import source.
                self._editor.doc.import_sources[import_source.uid] = (
                    import_source
                )

                # 2. Link the generated workpieces back to the source.
                all_new_workpieces: List[WorkPiece] = []
                for item in imported_items:
                    if isinstance(item, WorkPiece):
                        all_new_workpieces.append(item)
                    else:
                        all_new_workpieces.extend(
                            item.get_descendants(WorkPiece)
                        )
                for wp in all_new_workpieces:
                    wp.import_source_uid = import_source.uid

                # 3. Center and add them to the document in a transaction.
                self._fit_and_center_imported_items(imported_items)
                target_layer = cast(
                    Layer, self._editor.default_workpiece_layer
                )
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

        # Add the coroutine to the task manager with our safe callback.
        self._task_manager.add_coroutine(
            import_coro,
            key=f"import-{filename.name}",
            # when_done is wrapped to run on the main thread.
            when_done=commit_to_document,
        )

    def export_gcode_to_path(
        self, file_path: Path, when_done: Optional[Callable] = None
    ):
        """
        Headless version of G-code export that writes to a specific path.
        This is used by the async facade on DocEditor and for testing.
        """

        def write_gcode_sync(path, gcode):
            """Blocking I/O function to be run in a thread."""
            with open(path, "w", encoding="utf-8") as f:
                f.write(gcode)

        async def export_coro(context: ExecutionContext):
            machine = self._config_manager.config.machine
            if not machine:
                context.set_message("Error: No machine configured.")
                raise ValueError("Cannot export G-code without a machine.")

            try:
                # 1. Generate Ops (async, reports progress)
                ops = await generate_job_ops(
                    self._editor.doc,
                    machine,
                    self._editor.ops_generator,
                    context,
                )

                # 2. Encode G-code (sync, but usually fast)
                context.set_message(_("Encoding G-code..."))
                encoder = GcodeEncoder.for_machine(machine)
                gcode = encoder.encode(ops, machine, self._editor.doc)

                # 3. Write to file (sync, potentially slow, run in thread)
                context.set_message(_(f"Saving to {file_path}..."))
                await asyncio.to_thread(write_gcode_sync, file_path, gcode)

                context.set_message(_("Export complete!"))
                context.set_progress(1.0)
                context.flush()

            except Exception:
                logger.error("Failed to export G-code", exc_info=True)
                raise  # Re-raise to be caught by the task manager

        # Add the coroutine to the task manager
        self._task_manager.add_coroutine(
            export_coro, key="export-gcode", when_done=when_done
        )
